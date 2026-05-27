# Dataset sintetico — `src/make_dataset.py`

Genera `data/transactions.csv`: un dataset di transazioni bancarie finto ma
strutturato, pensato per fare *smoke-test* del modello su CPU. Le colonne sono
esattamente quelle attese dallo schema `DEFAULT_FEATURES` di `src/model.py`:

```
client_id, timestamp, importo, saldo_post, merchant,
mcc, canale, macro_tipo, sotto_tipo, divisa
```

---

## 1. L'idea

Il dataset non è rumore casuale: ha **struttura sufficiente perché il modello
possa davvero imparare qualcosa**, su tre assi che corrispondono ai tre obiettivi
di pre-training.

* **Cluster di clienti.** Ogni cliente appartiene a uno di pochi *tipi*
  (`CLIENT_TYPES`). I tipi spendono ad orari diversi, importi diversi, presso
  merchant diversi e con stipendi diversi → cluster separabili per la testa
  **contrastiva (InfoNCE)**.
* **Fingerprint per-cliente.** Dentro il suo tipo, ogni cliente ha un
  sottoinsieme stabile di merchant preferiti e un livello di spesa tipico
  suo → l'identità del singolo cliente è recuperabile.
* **Mapping merchant→categoria coerente.** Ogni merchant ha un profilo fisso di
  campi categorici (`mcc`, `macro_tipo`, `sotto_tipo`, `canale`): il merchant
  *predice* gli altri campi → la testa **MTM** vede segnale invece di rumore.
* **Pattern temporali.** Ritmo giornaliero (orari di spesa specifici del
  cluster) + uno **stipendio mensile ricorrente** in un giorno del mese fisso per
  cliente → l'encoder time-aware ha un pattern reale da modellare.

### Il knob della difficoltà: `noise_level`

Un singolo parametro `noise_level` (0.0 = pulito/separabile … 1.0 = molto
rumoroso) regola **tutte le fonti di rumore insieme**, lungo tutti e tre gli assi:

| Asse | Cosa succede alzando `noise_level` |
|------|-----------------------------------|
| **MTM** | il mapping merchant→categoria smette di essere deterministico (`mcc`/`sotto_tipo`/`canale`/`macro_tipo` prendono ogni tanto valori casuali in-vocab) → la testa deve modellare una *distribuzione* invece di memorizzare una lookup table. |
| **InfoNCE** | i cluster diventano meno separabili (overlap di merchant tra cluster, transazioni "off-pattern", maggiore varianza intra-cliente) → l'identità del cliente è più difficile da recuperare. |
| **Temporale** | il giorno dello stipendio oscilla, alcuni mesi vengono saltati, l'importo varia di più, compaiono rimborsi/storni occasionali. |

`noise_level=0.0` riproduce esattamente il comportamento pulito originale.

---

## 2. I parametri (come variare le distribuzioni)

### 2a. Costanti globali (in testa al file)

| Parametro | Default | Effetto |
|-----------|---------|---------|
| `N_TRANSACTIONS` | `400_000` | numero totale di righe generate |
| `N_CLIENTS` | `4_000` | numero di clienti su cui distribuire le transazioni |
| `MIN_N_TRANSACTIONS_PER_CLIENT` | `50` | floor di transazioni per cliente (clamp dopo lo split) |
| `NOISE_LEVEL` | `0.3` | difficoltà di default del task |
| `TS_BASE` | `1_577_836_800` | epoch di partenza (2020-01-01 00:00 UTC) |
| `TS_RANGE` | `~4 anni` | finestra entro cui cade lo start di ogni cliente |
| `DAY`, `MONTH` | — | costanti temporali (secondi) |

### 2b. Argomenti di `generate(...)`

```python
generate(n_transactions=N_TRANSACTIONS, n_clients=N_CLIENTS,
         out_path=DEFAULT_OUT, seed=0, noise_level=NOISE_LEVEL)
```

`seed` rende la generazione riproducibile; `out_path` decide dove scrivere il CSV.

### 2c. Definizione dei cluster — `CLIENT_TYPES`

È qui che si plasmano le distribuzioni "per tipo di cliente". Ogni voce è un dict:

| Chiave | Significato |
|--------|-------------|
| `name` | etichetta del cluster (es. `famiglia_giorno`) |
| `weight` | quota di clienti nel cluster (normalizzata internamente) |
| `hours` | curva oraria 24h dei consumi (vedi `_hours`) |
| `amount_mu` | media (in scala log) dell'importo speso — input a `lognormal` |
| `gap_days` | gap medio in giorni tra transazioni (scala dell'esponenziale) |
| `salary_mu` | media (log) dello stipendio mensile |
| `merchants` | pool di merchant da cui pesca il cluster |

I quattro cluster di default: `famiglia_giorno` (35%, spesa a pranzo),
`giovane_sera` (30%, sera), `altospendente` (20%, sera, importi alti),
`mattiniero_utenze` (15%, mattina).

**Per aggiungere/modificare un cluster** basta editare questa lista: i pesi
vengono rinormalizzati e la media globale di spesa (`_GLOBAL_AMOUNT_MU`, usata per
le transazioni off-pattern) si ricalcola da sola.

### 2d. Forma delle curve orarie — `_hours(peak, width=4.0)`

Gaussiana sulle 24 ore centrata su `peak`, più un floor `0.02`, normalizzata.
`width` regola quanto è concentrata la fascia oraria. Le tre curve predefinite:
`HOURS_MORNING` (9), `HOURS_MIDDAY` (13), `HOURS_EVENING` (20).

### 2e. Pool di merchant e loro profilo categorico

`MERCHANT_POOL` = `GROCERIES + SHOPPING + TRAVEL + UTILITIES + PAYMENTS`.
`MERCHANT_PROFILE` (costruito con un RNG a seed fisso `42`) assegna a ogni
merchant un `(mcc, macro_tipo, sotto_tipo, canale)` **fisso** entro i range del
vocabolario degli embedding (0 = padding). È questo che rende i campi categorici
correlati al merchant.

### 2f. Le probabilità di rumore (scalate da `noise_level`)

Sono derivate linearmente da `noise_level` dentro `_generate_client` / `_row`:

| Variabile | Formula | Cosa controlla |
|-----------|---------|----------------|
| `p_offpattern` | `0.2 · noise` | prob. che una tx ignori il fingerprint del cliente |
| `p_global_merchant` | `0.3 · noise` | prob. di pescare un merchant dall'intero pool (overlap tra cluster) |
| `p_refund` | `0.05 · noise` | prob. di un rimborso parziale dopo una spesa |
| spread `amount_mu` | `0.3 + 0.4 · noise` | varianza intra-cluster del livello di spesa |
| blend orario | `0.4 · noise` verso uniforme | quanto si "appiattisce" la curva oraria |
| `day_jitter` | `round(3 · noise)` | oscillazione del giorno-stipendio |
| `amt_sigma` | `0.02 + 0.1 · noise` | variabilità dell'importo dello stipendio |
| skip stipendio | `0.1 · noise` | prob. di saltare un mese di stipendio |
| `p_noise` per campo (in `_row`) | `mcc 0.5`, `canale 0.3`, `macro_tipo 0.15`, `sotto_tipo 0.5` × `noise` | prob. che il campo categorico venga randomizzato |
| `p_noneur` | `0.03 + 0.15 · noise` | prob. di una divisa ≠ EUR |

---

## 3. La logica (flusso di generazione)

### 3a. `generate()` — orchestrazione

1. **Clip** di `noise_level` in `[0,1]` e creazione dell'RNG con `seed`.
2. **Distribuzione delle transazioni sui clienti**: pesi `dirichlet(α=1.5)` →
   `multinomial` (skew tipo Zipf, alcuni clienti molto più attivi). Poi
   `clip` al minimo `MIN_N_TRANSACTIONS_PER_CLIENT` e aggiustamento ±1 finché la
   somma torna esattamente `n_transactions`.
3. **Assegnazione del cluster** a ogni cliente via `rng.choice(p=_TYPE_WEIGHTS)`.
4. Per ogni cliente chiama `_generate_client(...)` e accumula le righe.
5. **DataFrame**, ordinamento per `(client_id, timestamp)`, scrittura CSV e stampa
   di un riepilogo (righe, tx/cliente min/max/media, numerosità per cluster).

### 3b. `_generate_client()` — un cliente

1. **Fingerprint**: sottoinsieme di `fav_merchants` (3–7 merchant dal pool del
   cluster), `amount_mu` perturbato (spread cresce col rumore), saldo iniziale
   `lognormal(8.0, 0.8)`.
2. **Curva oraria** del cliente = mix tra quella del cluster e l'uniforme
   (più rumore ⇒ più uniforme).
3. **Stream di spesa**: start casuale nei primi ~2 anni; i gap tra transazioni
   sono `exponential(scale = gap_days·DAY)`; l'ora del giorno è campionata dalla
   curva oraria; il timestamp finale = giorno·DAY + ora·3600 + secondi.
4. **Per ogni transazione** (`_row`):
   * con prob. `p_offpattern` usa la media di spesa *globale* invece del
     fingerprint;
   * segno: ~85% spese (negativo), ~15% accrediti (positivo); importo
     `lognormal(mean=mu, sigma=1.0)`;
   * merchant: se off-pattern o con prob. `p_global_merchant` pesca dall'intero
     pool, altrimenti 70% dai preferiti / 30% dal pool del cluster;
   * eventuale **rimborso**: dopo una spesa, con prob. `p_refund`, un accredito
     parziale (30–100%) dello stesso merchant 1–2 giorni dopo. Essendo un credito
     non abbassa mai il saldo, quindi l'invariante `saldo_post ≥ 0` regge.
5. **Stipendio mensile**: giorno fisso `salary_day` (1–27), importo
   `lognormal(salary_mu, 0.2)`; per ogni mese tra prima e ultima transazione,
   con jitter sul giorno, possibile skip del mese, importo variabile; merchant
   fisso `"Datore"` con profilo categorico pinnato (`mcc=799, canale=4,
   macro_tipo=8, sotto_tipo=39`), pagato verso le 09:00.
6. **Roll-up del saldo**: si riordina tutto cronologicamente e si accumula
   `balance`. Una spesa viene **cappata ai fondi disponibili** (`importo` viene
   riscritto) così `saldo_post` resta sempre ≥ 0, come un conto reale.

### 3c. `_row()` — una riga

Costruisce il dict della transazione (`saldo_post` riempito dopo). I campi
categorici vengono dal `MERCHANT_PROFILE`, ma con prob. `p_noise·noise_level`
ciascuno viene **sostituito da un valore casuale in-vocab** → rompe il mapping
deterministico per la testa MTM. I campi **pinnati via `override`** (es. il
profilo dello stipendio) non vengono mai perturbati. `divisa` è EUR (`1`) salvo
prob. `p_noneur` in cui prende un codice valuta diverso (`2–5`).

---

## Uso

```bash
uv run python -m src.make_dataset
```

Modifica i default in testa al file, oppure chiama `generate(...)` con parametri
espliciti (utile per variare `noise_level` e `seed` da uno script).
```

**Convenzione padding** (coerente con lo schema del modello): indice/valore `0`
indica padding per categorici e numerici; il generatore produce valori sempre
`≥ 1` per i categorici, lasciando lo `0` libero per il padding a valle.
