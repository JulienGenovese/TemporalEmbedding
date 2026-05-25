"""Generate a small synthetic transaction dataset for CPU smoke-testing.

Deliberately simple, but with enough structure for the model to actually learn:

  * **Client clusters** — every client belongs to one of a few *types* that
    spend at different hours, in different amounts, on different merchants
    (plus a different salary). Clear, separable clusters for the contrastive
    head, and a real temporal pattern for the time-aware encoder.
  * **Per-client fingerprint** — within its type, each client favours a stable
    merchant subset and has its own typical spend level.
  * **Coherent merchant→MCC mapping** — categorical fields are correlated, so
    the MTM head sees signal instead of noise.
  * **Temporal patterns** — daily rhythm (cluster-specific spending hours) plus
    a recurring monthly salary on a fixed per-client day.

A single ``noise_level`` knob (0.0 = clean/separable, 1.0 = very noisy) tunes how
hard the pre-training task is along all three axes at once:

  * **MTM** — the merchant→category mapping stops being deterministic (mcc /
    sotto_tipo / canale / macro_tipo get occasional random values), so the head
    must model a distribution instead of memorising a lookup table.
  * **InfoNCE** — clusters become less separable (cross-cluster merchant overlap,
    off-pattern transactions, larger intra-client variance), so client identity
    is harder to recover.
  * **Temporal** — salary day jitters, some months are skipped, the salary amount
    varies more, and occasional refunds/reversals appear.

``noise_level=0.0`` reproduces the original clean behaviour exactly.

Produces ``data/transactions.csv`` with every column expected by the
``DEFAULT_FEATURES`` schema in :mod:`src.model`:

    client_id, timestamp, importo, saldo_post, merchant,
    mcc, canale, macro_tipo, sotto_tipo, divisa

Usage:
    uv run python -m src.make_dataset
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRANSACTIONS = 400_000
N_CLIENTS = 4_000
MIN_N_TRANSACTIONS_PER_CLIENT = 50
NOISE_LEVEL = 0.3                  # default difficulty knob (0=clean … 1=very noisy)
TS_BASE = 1_577_836_800            # 2020-01-01 00:00 UTC
TS_RANGE = 4 * 365 * 24 * 3600     # ~4 years in seconds
DAY = 86_400
MONTH = 30 * DAY


def _hours(peak: int, width: float = 4.0) -> np.ndarray:
    """A normalised 24-hour weight curve peaked around ``peak`` o'clock."""
    h = np.arange(24)
    w = np.exp(-0.5 * ((h - peak) / width) ** 2) + 0.02
    return w / w.sum()


HOURS_MORNING = _hours(9)      # commuters / utilities
HOURS_MIDDAY  = _hours(13)     # families doing groceries
HOURS_EVENING = _hours(20)     # young / dining / shopping


# Merchant themes — used to give each client type a distinct merchant pool.
GROCERIES = ["Esselunga", "Conad", "Carrefour", "Lidl", "Aldi", "Coop"]
SHOPPING  = ["Amazon", "Walmart", "IKEA", "Apple", "Zara", "H&M",
             "Decathlon", "Mediaworld", "Unieuro"]
TRAVEL    = ["Trenitalia", "Booking", "Airbnb", "RyanAir", "Easyjet"]
UTILITIES = ["TIM", "Vodafone", "Eni", "Shell"]
PAYMENTS  = ["PayPal", "Satispay", "Bancomat", "Netflix", "Spotify", "Starbucks"]
MERCHANT_POOL = GROCERIES + SHOPPING + TRAVEL + UTILITIES + PAYMENTS


# Each merchant gets a fixed (mcc, macro_tipo, sotto_tipo, canale) profile so
# the categorical fields are correlated (the merchant predicts the rest). Values
# stay inside the embedding vocab ranges (0 = padding).
_rng0 = np.random.default_rng(42)
MERCHANT_PROFILE: dict[str, dict[str, int]] = {
    m: {
        "mcc":        int(_rng0.integers(1, 800)),
        "macro_tipo": int(_rng0.integers(1, 8)),
        "sotto_tipo": int(_rng0.integers(1, 40)),
        "canale":     int(_rng0.integers(1, 10)),
    }
    for m in MERCHANT_POOL
}


# Client clusters: each spends at different hours, in different amounts, on a
# different merchant pool, with a different salary. ``weight`` is the share of
# clients in the cluster.
CLIENT_TYPES: list[dict] = [
    {"name": "famiglia_giorno",  "weight": 0.35, "hours": HOURS_MIDDAY,
     "amount_mu": 3.4, "gap_days": 1.5, "salary_mu": 7.7,
     "merchants": GROCERIES + PAYMENTS + UTILITIES},
    {"name": "giovane_sera",     "weight": 0.30, "hours": HOURS_EVENING,
     "amount_mu": 2.8, "gap_days": 2.0, "salary_mu": 7.2,
     "merchants": PAYMENTS + SHOPPING + ["Starbucks"]},
    {"name": "altospendente",    "weight": 0.20, "hours": HOURS_EVENING,
     "amount_mu": 4.2, "gap_days": 3.0, "salary_mu": 8.5,
     "merchants": SHOPPING + TRAVEL},
    {"name": "mattiniero_utenze","weight": 0.15, "hours": HOURS_MORNING,
     "amount_mu": 3.3, "gap_days": 2.5, "salary_mu": 7.9,
     "merchants": UTILITIES + GROCERIES},
]
_TYPE_WEIGHTS = np.array([t["weight"] for t in CLIENT_TYPES])
_TYPE_WEIGHTS = _TYPE_WEIGHTS / _TYPE_WEIGHTS.sum()

# Population-average spend level — used for "off-pattern" transactions that don't
# follow the client's own fingerprint (adds intra-client variance under noise).
_GLOBAL_AMOUNT_MU = float(np.mean([t["amount_mu"] for t in CLIENT_TYPES]))
_UNIFORM_HOURS = np.full(24, 1.0 / 24)

DEFAULT_OUT = Path("data") / "transactions.csv"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(
    n_transactions: int = N_TRANSACTIONS,
    n_clients: int = N_CLIENTS,
    out_path: Path | str = DEFAULT_OUT,
    seed: int = 0,
    noise_level: float = NOISE_LEVEL,
) -> Path:
    """Generate the synthetic CSV.

    ``noise_level`` (0.0–1.0) scales every noise source linearly: 0.0 reproduces
    the original clean/separable dataset, higher values make the MTM, InfoNCE and
    temporal tasks progressively harder.
    """
    noise_level = float(np.clip(noise_level, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    # Distribute transactions across clients with a mild Zipf-like skew.
    weights = rng.dirichlet(np.ones(n_clients) * 1.5)
    counts = rng.multinomial(n_transactions, weights)
    counts = np.clip(counts, MIN_N_TRANSACTIONS_PER_CLIENT, None)
    while counts.sum() > n_transactions:
        counts[counts.argmax()] -= 1
    while counts.sum() < n_transactions:
        counts[counts.argmin()] += 1

    # Assign each client to a cluster.
    type_idx = rng.choice(len(CLIENT_TYPES), size=n_clients, p=_TYPE_WEIGHTS)

    rows = []
    for client_id, n in enumerate(counts):
        ctype = CLIENT_TYPES[type_idx[client_id]]
        rows.extend(_generate_client(client_id, int(n), ctype, rng, noise_level))

    df = pd.DataFrame(rows, columns=[
        "client_id", "timestamp", "importo", "saldo_post", "merchant",
        "mcc", "canale", "macro_tipo", "sotto_tipo", "divisa",
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


def _generate_client(client_id: int, n_tx: int, ctype: dict,
                     rng: np.random.Generator, noise_level: float = 0.0) -> list[dict]:
    """Generate ``n_tx`` transactions for one client of cluster ``ctype``.

    Temporal patterns: a cluster-specific daily rhythm on spending plus a
    recurring monthly salary on a fixed day-of-month. ``noise_level`` (0–1)
    scales the cross-cluster merchant overlap, off-pattern transactions, larger
    fingerprint variance, salary irregularity and occasional refunds.
    """
    # Per-client fingerprint within its cluster. A larger noise_level widens the
    # intra-cluster spread, so clients of the same type look less alike.
    pool = ctype["merchants"]
    fav_merchants = rng.choice(pool, size=min(rng.integers(3, 8), len(pool)), replace=False)
    amount_mu = ctype["amount_mu"] + rng.normal(0, 0.3 + 0.4 * noise_level)
    balance = float(rng.lognormal(8.0, 0.8))

    # Cluster daily rhythm, blended toward uniform as noise rises (fuzzier hours).
    hours_p = (1.0 - 0.4 * noise_level) * ctype["hours"] + (0.4 * noise_level) * _UNIFORM_HOURS
    hours_p = hours_p / hours_p.sum()

    p_offpattern = 0.2 * noise_level    # transaction ignores the client fingerprint
    p_global_merchant = 0.3 * noise_level  # merchant drawn from the whole pool
    p_refund = 0.05 * noise_level       # debit immediately followed by a partial refund

    # --- spending stream: cluster-specific cadence + daytime hour ---
    start = TS_BASE + int(rng.integers(0, TS_RANGE // 2))
    gaps = rng.exponential(scale=ctype["gap_days"] * DAY, size=n_tx).astype(np.int64)
    days = (start + np.cumsum(gaps)) // DAY
    hours = rng.choice(24, size=n_tx, p=hours_p)            # cluster's daily rhythm
    secs = rng.integers(0, 3600, size=n_tx)
    timestamps = days * DAY + hours * 3600 + secs

    rows: list[dict] = []
    for ts in timestamps:
        off = rng.random() < p_offpattern
        mu = _GLOBAL_AMOUNT_MU if off else amount_mu        # off-pattern ignores fingerprint
        sign = -1 if rng.random() < 0.85 else 1             # ~85% spese, ~15% accrediti
        amount = sign * float(rng.lognormal(mean=mu, sigma=1.0))
        if off or rng.random() < p_global_merchant:
            merchant = str(rng.choice(MERCHANT_POOL))       # cross-cluster overlap
        else:
            merchant = (str(rng.choice(fav_merchants)) if rng.random() < 0.7
                        else str(rng.choice(pool)))
        rows.append(_row(client_id, int(ts), amount, merchant, rng, noise_level))

        # Occasional refund/reversal: a positive credit shortly after a debit,
        # same merchant, partial amount. Being a credit it never lowers the
        # balance, so the >=0 invariant in the roll-up below is preserved.
        if sign < 0 and rng.random() < p_refund:
            refund_ts = int(ts) + int(rng.integers(1, 3) * DAY)
            refund_amt = -amount * float(rng.uniform(0.3, 1.0))   # amount<0 → credit
            rows.append(_row(client_id, refund_ts, refund_amt, merchant, rng, noise_level))

    # --- recurring monthly salary; day jitters and some months are skipped ---
    salary_day = int(rng.integers(1, 28))
    salary_amt = float(rng.lognormal(ctype["salary_mu"], 0.2))
    day_jitter = int(round(3 * noise_level))
    amt_sigma = 0.02 + 0.1 * noise_level
    first, last = int(timestamps.min()), int(timestamps.max())
    month_start = (first // MONTH) * MONTH
    for base in range(month_start, last + MONTH, MONTH):
        if rng.random() < 0.1 * noise_level:               # skipped paycheck
            continue
        day = salary_day
        if day_jitter:
            day = int(np.clip(salary_day + rng.integers(-day_jitter, day_jitter + 1), 1, 27))
        ts = base + day * DAY + 9 * 3600                    # paid ~09:00
        amt = salary_amt * (1.0 + rng.normal(0, amt_sigma))
        rows.append(_row(client_id, ts, amt, "Datore", rng, noise_level, mcc=799,
                         canale=4, macro_tipo=8, sotto_tipo=39))

    # --- sort chronologically and roll up the running balance ---
    # The balance can never go negative: a spesa is capped at the funds
    # available, so ``importo`` stays consistent with ``saldo_post`` (which
    # therefore is always >= 0, as a real account balance).
    rows.sort(key=lambda r: r["timestamp"])
    for r in rows:
        amount = r["importo"]
        if balance + amount < 0:
            amount = -balance              # spend at most what is available
            r["importo"] = round(amount, 2)
        balance += amount
        r["saldo_post"] = round(balance, 2)
    return rows


def _row(client_id: int, ts: int, amount: float, merchant: str,
         rng: np.random.Generator, noise_level: float = 0.0, **override: int) -> dict:
    """Build one transaction row; ``saldo_post`` is filled in later.

    With ``noise_level > 0`` the merchant→category mapping is no longer
    deterministic: each categorical field is occasionally replaced by a random
    in-vocab value, so the MTM head must model a distribution rather than a
    lookup table. Fields pinned via ``override`` (e.g. the salary profile) are
    never perturbed.
    """
    prof = MERCHANT_PROFILE.get(merchant, {})

    def _cat(name: str, default: int, p_noise: float, lo: int, hi: int) -> int:
        if name in override:                       # pinned value → never perturbed
            return override[name]
        if noise_level and rng.random() < p_noise * noise_level:
            return int(rng.integers(lo, hi))       # random in-vocab (hi exclusive)
        return prof.get(name, default)

    p_noneur = 0.03 + 0.15 * noise_level
    return {
        "client_id":  client_id,
        "timestamp":  int(ts),
        "importo":    round(float(amount), 2),
        "saldo_post": 0.0,
        "merchant":   merchant,
        "mcc":        _cat("mcc",        1, 0.5,  1, 801),
        "canale":     _cat("canale",     1, 0.3,  1,  11),
        "macro_tipo": _cat("macro_tipo", 1, 0.15, 1,   9),
        "sotto_tipo": _cat("sotto_tipo", 1, 0.5,  1,  41),
        "divisa":     1 if rng.random() < (1.0 - p_noneur) else int(rng.integers(2, 6)),
    }


if __name__ == "__main__":
    generate()
