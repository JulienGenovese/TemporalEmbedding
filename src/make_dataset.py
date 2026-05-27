"""Genera un piccolo dataset sintetico di transazioni per smoke-test su CPU.

Volutamente semplice, ma con abbastanza struttura perché il modello impari
davvero qualcosa:

  * **Cluster di clienti** — ogni cliente appartiene a uno di pochi *tipi* che
    spendono a ore diverse, per importi diversi, presso esercenti diversi.
    Cluster netti e separabili per la testa contrastiva (InfoNCE) e un vero
    pattern temporale per l'encoder time-aware.
  * **Impronta per-cliente** — all'interno del suo tipo, ogni cliente predilige
    un sottoinsieme stabile di esercenti e ha un proprio livello di spesa tipico.
  * **Mappatura coerente esercente→MCC** — i campi categorici sono correlati
    (l'esercente predice gli altri), così la testa MTM vede segnale e non rumore.
  * **Pattern temporali** — un ritmo di spesa giornaliero specifico del cluster.

COME VIENE GENERATO IL DATASET (vista d'insieme delle distribuzioni usate)

  1. *Quante transazioni per cliente?*  Una **Dirichlet(α=1.5)** genera i pesi
     dei clienti e una **Multinomiale** distribuisce le N transazioni su quei
     pesi. La Dirichlet con α<1 (qui 1.5, leggermente >1 ma comunque "morbida")
     produce pesi disomogenei → alcuni clienti molto attivi, molti poco attivi,
     come nella realtà (distribuzione a coda lunga del numero di transazioni).
  2. *A che tipo appartiene il cliente?*  Estrazione **categorica** sui
     ``_TYPE_WEIGHTS`` (quote dei cluster).
  3. *Quando avvengono le transazioni?*  I gap fra transazioni consecutive sono
     **Esponenziali** (processo di arrivo tipo Poisson → tempi di attesa
     esponenziali); l'ora del giorno è **categorica** pesata da una **gaussiana**
     centrata sull'ora di punta del cluster; i secondi sono **uniformi**.
  4. *Di che importo?*  L'importo è **Log-normale** (sempre positivo, asimmetrico
     a destra, code lunghe) — la scelta classica per modellare importi/redditi.
  5. *Quale esercente / categoria?*  Estrazioni **categoriche** dal pool del
     cluster, con una mappa fissa esercente→(mcc, macro_tipo).

Una sola manopola ``noise_level`` (0.0 = pulito/separabile, 1.0 = molto rumoroso)
regola contemporaneamente la difficoltà del pre-training sui tre assi:

  * **MTM** — la mappa esercente→categoria smette di essere deterministica (mcc /
    macro_tipo prendono ogni tanto valori casuali), quindi la testa deve
    modellare una *distribuzione* invece di memorizzare una lookup table.
  * **InfoNCE** — i cluster diventano meno separabili (sovrapposizione di
    esercenti fra cluster, transazioni fuori-pattern, maggiore varianza
    intra-cliente), quindi l'identità del cliente è più difficile da recuperare.
  * **Temporale** — compaiono occasionali rimborsi/storni poco dopo un addebito.

``noise_level=0.0`` riproduce esattamente il comportamento pulito originale.

Produce ``data/transactions.csv`` con tutte le colonne attese dallo schema
``DEFAULT_FEATURES`` in :mod:`src.model`:

    client_id, timestamp, importo, merchant, mcc, macro_tipo

Uso:
    uv run python -m src.make_dataset
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

N_TRANSACTIONS = 400_000
N_CLIENTS = 4_000
MIN_N_TRANSACTIONS_PER_CLIENT = 50
NOISE_LEVEL = 0.3                  # manopola di difficoltà di default (0=pulito … 1=molto rumoroso)
TS_BASE = 1_577_836_800            # 2020-01-01 00:00 UTC
TS_RANGE = 4 * 365 * 24 * 3600     # ~4 anni in secondi
DAY = 86_400


def _hours(peak: int, width: float = 4.0) -> np.ndarray:
    """Curva di pesi sulle 24 ore, a forma di **gaussiana** centrata su ``peak``.

    Modelliamo l'ora del giorno con una campana gaussiana perché il consumo reale
    è concentrato attorno a un'ora di punta e cala gradualmente ai lati (non è né
    uniforme né a gradino). ``width`` è la deviazione standard in ore. Aggiungiamo
    un fondo costante (+0.02) così nessuna ora ha probabilità nulla, poi
    normalizziamo per ottenere una distribuzione di probabilità valida (somma 1).
    """
    h = np.arange(24)
    w = np.exp(-0.5 * ((h - peak) / width) ** 2) + 0.02
    return w / w.sum()


HOURS_MORNING = _hours(9)      # pendolari / utenze
HOURS_MIDDAY  = _hours(13)     # famiglie che fanno la spesa
HOURS_EVENING = _hours(20)     # giovani / cene / shopping


# Temi di esercenti — servono a dare a ogni tipo di cliente un pool distinto.
GROCERIES = ["Esselunga", "Conad", "Carrefour", "Lidl", "Aldi", "Coop"]
SHOPPING  = ["Amazon", "Walmart", "IKEA", "Apple", "Zara", "H&M",
             "Decathlon", "Mediaworld", "Unieuro"]
TRAVEL    = ["Trenitalia", "Booking", "Airbnb", "RyanAir", "Easyjet"]
UTILITIES = ["TIM", "Vodafone", "Eni", "Shell"]
PAYMENTS  = ["PayPal", "Satispay", "Bancomat", "Netflix", "Spotify", "Starbucks"]
MERCHANT_POOL = GROCERIES + SHOPPING + TRAVEL + UTILITIES + PAYMENTS


# A ogni esercente assegniamo un profilo fisso (mcc, macro_tipo) così i campi
# categorici sono correlati (l'esercente predice il resto). I valori restano
# dentro i range del vocabolario degli embedding (0 = padding). ``macro_tipo``
# copre un insieme più ampio di categorie grossolane (1..24).
# Estrazione **uniforme discreta** (rng.integers): senza informazioni a priori
# sull'MCC "giusto", ogni codice in-vocab è a priori equiprobabile; ciò che conta
# è che la mappa sia *deterministica e stabile* per esercente, non come è scelta.
# Il seed fisso (42) garantisce che il profilo sia identico fra esecuzioni.
_rng0 = np.random.default_rng(42)
MERCHANT_PROFILE: dict[str, dict[str, int]] = {
    m: {
        "mcc":        int(_rng0.integers(1, 800)),
        "macro_tipo": int(_rng0.integers(1, 24)),
    }
    for m in MERCHANT_POOL
}


# Cluster di clienti: ognuno spende a ore diverse, per importi diversi, su un
# pool di esercenti diverso. ``weight`` è la quota di clienti nel cluster.
#   - hours:      curva oraria (gaussiana) del cluster
#   - amount_mu:  media (in spazio log) della log-normale degli importi
#   - gap_days:   media (in giorni) dell'esponenziale dei tempi fra transazioni
CLIENT_TYPES: list[dict] = [
    {"name": "famiglia_giorno",
     "weight": 0.35,
     "hours": HOURS_MIDDAY,
     "amount_mu": 3.4,
     "gap_days": 1.5,
     "merchants": GROCERIES + PAYMENTS + UTILITIES},

    {"name": "giovane_sera",
     "weight": 0.30,
     "hours": HOURS_EVENING,
     "amount_mu": 2.8,
     "gap_days": 2.0,
     "merchants": PAYMENTS + SHOPPING + ["Starbucks"]},

    {"name": "altospendente",
     "weight": 0.20,
     "hours": HOURS_EVENING,
     "amount_mu": 4.2,
     "gap_days": 3.0,
     "merchants": SHOPPING + TRAVEL},

    {"name": "mattiniero_utenze",
     "weight": 0.15,
     "hours": HOURS_MORNING,
     "amount_mu": 3.3,
     "gap_days": 2.5,
     "merchants": UTILITIES + GROCERIES},
]
_TYPE_WEIGHTS = np.array([t["weight"] for t in CLIENT_TYPES])
_TYPE_WEIGHTS = _TYPE_WEIGHTS / _TYPE_WEIGHTS.sum()

# Livello di spesa medio della popolazione — usato per le transazioni
# "fuori-pattern" che non seguono l'impronta del cliente (aggiunge varianza
# intra-cliente quando il rumore cresce).
_GLOBAL_AMOUNT_MU = float(np.mean([t["amount_mu"] for t in CLIENT_TYPES]))
_UNIFORM_HOURS = np.full(24, 1.0 / 24)

DEFAULT_OUT = Path("data") / "transactions.csv"


# ---------------------------------------------------------------------------
# Generatore
# ---------------------------------------------------------------------------

def generate(
    n_transactions: int = N_TRANSACTIONS,
    n_clients: int = N_CLIENTS,
    out_path: Path | str = DEFAULT_OUT,
    seed: int = 0,
    noise_level: float = NOISE_LEVEL,
) -> Path:
    """Genera il CSV sintetico.

    ``noise_level`` (0.0–1.0) scala linearmente ogni sorgente di rumore: 0.0
    riproduce il dataset pulito/separabile originale, valori più alti rendono i
    task MTM, InfoNCE e temporale progressivamente più difficili.

    Passi:
      1. Quante transazioni a testa → **Dirichlet** + **Multinomiale**.
      2. Tipo (cluster) di ogni cliente → estrazione **categorica**.
      3. Per ogni cliente, genera le sue transazioni (vedi ``_generate_client``).
    """
    noise_level = float(np.clip(noise_level, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    # --- Quante transazioni per cliente ---
    # Dirichlet(α=1.5) → un vettore di pesi che somma a 1, con disomogeneità
    # controllata (α più piccolo ⇒ più disomogeneo). La Multinomiale distribuisce
    # poi le n_transactions su quei pesi: il risultato è una coda lunga realistica
    # (pochi clienti molto attivi, molti poco attivi).
    weights = rng.dirichlet(np.ones(n_clients) * 1.5)
    counts = rng.multinomial(n_transactions, weights)
    # Imponiamo un minimo per cliente, poi riaggiustiamo per conservare il totale
    # esatto (togliamo dai più grandi / aggiungiamo ai più piccoli).
    counts = np.clip(counts, MIN_N_TRANSACTIONS_PER_CLIENT, None)
    while counts.sum() > n_transactions:
        counts[counts.argmax()] -= 1
    while counts.sum() < n_transactions:
        counts[counts.argmin()] += 1

    # --- Assegna ogni cliente a un cluster (estrazione categorica sui pesi) ---
    type_idx = rng.choice(len(CLIENT_TYPES), size=n_clients, p=_TYPE_WEIGHTS)

    rows = []
    for client_id, n in enumerate(counts):
        ctype = CLIENT_TYPES[type_idx[client_id]]
        rows.extend(_generate_client(client_id, int(n), ctype, rng, noise_level))

    df = pd.DataFrame(rows, columns=[
        "client_id", "timestamp", "importo", "merchant",
        "mcc", "macro_tipo",
    ])
    df = df.sort_values(["client_id", "timestamp"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    per_client = df.groupby("client_id").size()
    print(f"Saved {len(df):,} rows × {len(df.columns)} cols → {out_path} "
          f"(noise_level={noise_level:.2f})")
    print(f"  clients: {df['client_id'].nunique()} "
          f"(min={per_client.min()}, max={per_client.max()}, mean={per_client.mean():.1f} tx/client)")
    print("  clusters: " + ", ".join(
        f"{t['name']}={int((type_idx == i).sum())}" for i, t in enumerate(CLIENT_TYPES)))
    return out_path


def _client_fingerprint(ctype: dict, rng: np.random.Generator,
                        noise_level: float) -> tuple[np.ndarray, float]:
    """Impronta del singolo cliente all'interno del suo cluster.

    Restituisce ``(fav_merchants, amount_mu)``: un sottoinsieme stabile di
    esercenti preferiti e un livello di spesa tipico.

    Distribuzioni usate:
      * ``fav_merchants`` — quanti preferiti: **uniforme discreta** in [3,7]
        (``rng.integers(3, 8)``); *quali*: campionamento **uniforme senza
        rimpiazzo** dal pool del cluster (``replace=False``), così i preferiti
        sono distinti.
      * ``amount_mu`` — la media (in spazio log) della log-normale degli importi
        è il valore del cluster più una perturbazione **gaussiana** N(0, σ). La
        σ cresce con il rumore (0.3 → 0.7), allargando la dispersione *fra*
        clienti dello stesso tipo: con più rumore due clienti dello stesso
        cluster si somigliano meno (InfoNCE più difficile).
    """
    pool = ctype["merchants"]
    fav_merchants = rng.choice(pool, size=min(rng.integers(3, 8), len(pool)), replace=False)
    amount_mu = ctype["amount_mu"] + rng.normal(0, 0.3 + 0.4 * noise_level)
    return fav_merchants, amount_mu


def _generate_timestamps(n_tx: int, ctype: dict, rng: np.random.Generator,
                         noise_level: float) -> np.ndarray:
    """Timestamp Unix in secondi (giorni · ore · secondi) per un cliente.

    Costruiamo il timestamp da tre componenti, ciascuna con la sua distribuzione:

      * **GIORNI — gap esponenziali.** I tempi di attesa fra transazioni
        consecutive seguono una **Esponenziale** con media ``gap_days`` (in
        giorni). È la scelta naturale per gli "inter-arrival time" di un processo
        di arrivo tipo Poisson (eventi senza memoria); produce molte transazioni
        ravvicinate e qualche pausa lunga, come la spesa reale. Si parte da un
        istante iniziale **uniforme** nei primi ~2 anni e si accumulano i gap
        (``cumsum``) → i timestamp sono crescenti per costruzione.
      * **ORE — categorica pesata da una gaussiana.** L'ora del giorno è estratta
        da una categorica su {0..23} con pesi ``hours_p``, cioè la campana
        gaussiana del cluster (vedi ``_hours``). Con il rumore mescoliamo questa
        campana verso l'**uniforme** (peso 0.4·noise): le ore diventano più
        sfocate e i cluster meno riconoscibili dal solo orario.
      * **SECONDI — uniformi.** Entro l'ora, i secondi sono **uniformi** in
        [0, 3600): non c'è un pattern al secondo, quindi l'uniforme è corretta.
    """
    # Mix campana-del-cluster ↔ uniforme: a noise=0 è pura campana, a noise=1
    # pesa l'uniforme per il 40%. Rinormalizziamo per riavere una distribuzione.
    hours_p = (1.0 - 0.4 * noise_level) * ctype["hours"] + (0.4 * noise_level) * _UNIFORM_HOURS
    hours_p = hours_p / hours_p.sum()

    start = TS_BASE + int(rng.integers(0, TS_RANGE // 2))          # inizio uniforme
    gaps = rng.exponential(scale=ctype["gap_days"] * DAY, size=n_tx).astype(np.int64)  # gap esponenziali
    days = (start + np.cumsum(gaps)) // DAY                        # giorni (cumulati)
    hours = rng.choice(24, size=n_tx, p=hours_p)                   # ora ~ campana del cluster
    secs = rng.integers(0, 3600, size=n_tx)                        # secondi uniformi nell'ora
    return days * DAY + hours * 3600 + secs


def _generate_amount(mu: float, rng: np.random.Generator) -> tuple[float, int]:
    """Un importo con segno, da una **log-normale**.

    Restituisce ``(amount, sign)`` con ~85% addebiti (spese, ``sign < 0``) e
    ~15% accrediti (``sign > 0``) — il segno è una **Bernoulli** (p=0.85).

    Perché log-normale: gli importi sono sempre positivi, fortemente asimmetrici
    a destra e con code lunghe (tante piccole spese, poche molto grandi). Una
    variabile X = exp(N(μ, σ²)) cattura esattamente questo: ``mu`` è la media nello
    spazio logaritmico (livello di spesa tipico del cluster/cliente), ``sigma``
    l'ampiezza. Il segno viene applicato dopo: l'importo è |X| con segno.
    """
    sign = -1 if rng.random() < 0.85 else 1
    return sign * float(rng.lognormal(mean=mu, sigma=1.0)), sign


def _generate_merchant(off: bool, pool: list, fav_merchants: np.ndarray,
                       rng: np.random.Generator, p_global_merchant: float) -> str:
    """Sceglie l'esercente di una transazione (estrazioni **categoriche/uniformi**).

    Logica gerarchica:
      * Se la transazione è *fuori-pattern* (``off``) oppure scatta un'estrazione
        "globale" (con prob. ``p_global_merchant``, che cresce col rumore), si
        pesca **uniformemente da tutto** ``MERCHANT_POOL`` → sovrapposizione fra
        cluster, identità del cliente più difficile (InfoNCE).
      * Altrimenti il cliente segue la sua impronta: con prob. 0.7 sceglie dai
        propri preferiti ``fav_merchants``, con prob. 0.3 da tutto il pool del
        cluster — sempre estrazioni **uniformi** sull'insieme scelto. Lo split
        0.7/0.3 è una **Bernoulli** che dà concentrazione sui preferiti ma non
        totale rigidità.
    """
    if off or rng.random() < p_global_merchant:
        return str(rng.choice(MERCHANT_POOL))               # sovrapposizione fra cluster
    return (str(rng.choice(fav_merchants)) if rng.random() < 0.7
            else str(rng.choice(pool)))


def _generate_refund(ts: int, amount: float,
                     rng: np.random.Generator) -> tuple[int, float]:
    """Un rimborso/storno parziale di un addebito.

    Restituisce ``(refund_ts, refund_amt)``: un accredito (``amount > 0``) 1–2
    giorni dopo l'addebito originale, per il 30–100% del suo valore.

    Distribuzioni:
      * ritardo in giorni — **uniforme discreta** in {1, 2} (``integers(1, 3)``);
      * frazione rimborsata — **uniforme continua** in [0.3, 1.0]. Non avendo un
        motivo per privilegiare un rimborso parziale specifico, l'uniforme è la
        scelta neutra. Il segno si inverte (addebito<0 → accredito>0).
    """
    refund_ts = int(ts) + int(rng.integers(1, 3) * DAY)
    refund_amt = -amount * float(rng.uniform(0.3, 1.0))     # amount<0 (addebito) → accredito
    return refund_ts, refund_amt


def _generate_client(client_id: int,
                     n_tx: int,
                     ctype: dict,
                     rng: np.random.Generator, noise_level: float = 0.0) -> list[dict]:
    """Genera ``n_tx`` transazioni per un cliente del cluster ``ctype``.

    Orchestra i generatori per-componente (:func:`_client_fingerprint`,
    :func:`_generate_timestamps`, :func:`_generate_amount`,
    :func:`_generate_merchant`, :func:`_generate_refund`). ``noise_level`` (0–1)
    scala la sovrapposizione di esercenti fra cluster, le transazioni
    fuori-pattern, la varianza dell'impronta e i rimborsi occasionali.

    Per ogni transazione le probabilità "di rumore" sono **Bernoulli** la cui p
    cresce linearmente col ``noise_level`` (a noise=0 sono tutte spente, quindi il
    dataset è pulito e perfettamente separabile).
    """
    pool = ctype["merchants"]
    fav_merchants, amount_mu = _client_fingerprint(ctype, rng, noise_level)
    timestamps = _generate_timestamps(n_tx, ctype, rng, noise_level)

    p_offpattern = 0.2 * noise_level       # la transazione ignora l'impronta del cliente
    p_global_merchant = 0.3 * noise_level  # esercente pescato da tutto il pool
    p_refund = 0.05 * noise_level          # addebito seguito da un rimborso parziale

    rows: list[dict] = []
    for ts in timestamps:
        off = rng.random() < p_offpattern
        mu = _GLOBAL_AMOUNT_MU if off else amount_mu        # fuori-pattern: ignora l'impronta
        amount, sign = _generate_amount(mu, rng)
        merchant = _generate_merchant(off, pool, fav_merchants, rng, p_global_merchant)
        rows.append(_row(client_id, int(ts), amount, merchant, rng, noise_level))

        if sign < 0 and rng.random() < p_refund:
            refund_ts, refund_amt = _generate_refund(ts, amount, rng)
            rows.append(_row(client_id, refund_ts, refund_amt, merchant, rng, noise_level))

    # --- ordina cronologicamente ---
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _row(client_id: int, ts: int, amount: float, merchant: str,
         rng: np.random.Generator, noise_level: float = 0.0, **override: int) -> dict:
    """Costruisce una riga-transazione.

    Con ``noise_level > 0`` la mappa esercente→categoria non è più deterministica:
    ogni campo categorico viene ogni tanto sostituito da un valore casuale
    in-vocab, così la testa MTM deve modellare una *distribuzione* invece di una
    lookup table. I campi fissati via ``override`` non vengono mai perturbati.

    Distribuzione del rumore categorico: una **Bernoulli** (p = ``p_noise`` ·
    ``noise_level``) decide *se* perturbare; in caso affermativo il nuovo valore è
    **uniforme discreto** nel range in-vocab [lo, hi). Si noti che ``mcc`` ha
    p_noise più alto (0.5) di ``macro_tipo`` (0.15): l'MCC, più granulare, è reso
    più rumoroso della categoria grossolana.
    """
    prof = MERCHANT_PROFILE.get(merchant, {})

    def _cat(name: str, default: int, p_noise: float, lo: int, hi: int) -> int:
        if name in override:                       # valore fissato → mai perturbato
            return override[name]
        if noise_level and rng.random() < p_noise * noise_level:
            return int(rng.integers(lo, hi))       # casuale in-vocab (hi escluso)
        return prof.get(name, default)

    return {
        "client_id":  client_id,
        "timestamp":  int(ts),
        "importo":    round(float(amount), 2),
        "merchant":   merchant,
        "mcc":        _cat("mcc",        1, 0.5,  1, 801),
        "macro_tipo": _cat("macro_tipo", 1, 0.15, 1,  25),
    }


if __name__ == "__main__":
    generate()
