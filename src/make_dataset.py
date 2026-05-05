"""Generate a small synthetic transaction dataset for CPU smoke-testing.

Produces ``data/transactions.csv`` with 10 000 rows distributed across
``N_CLIENTS`` clients, each row carrying every field expected by the
``DEFAULT_FEATURES`` schema in :mod:`src.model`.

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

N_TRANSACTIONS = 10_000
N_CLIENTS = 100

TS_BASE = 1_577_836_800            # 2020-01-01 00:00 UTC
TS_RANGE = 4 * 365 * 24 * 3600     # ~4 years in seconds

# Vocab sizes — kept slightly below the model defaults so every drawn ID
# falls inside the embedding tables when the schema is used as-is.
MCC_RANGE        = (1, 800)
CANALE_RANGE     = (1, 10)
MACRO_TIPO_RANGE = (1, 8)
SOTTO_TIPO_RANGE = (1, 40)
DIVISA_RANGE     = (1, 5)

MERCHANT_POOL = [
    "Amazon", "Walmart", "Starbucks", "Esselunga", "Conad",
    "IKEA", "Apple", "Netflix", "Spotify", "TIM",
    "Vodafone", "Eni", "Shell", "Trenitalia", "Decathlon",
    "Zara", "H&M", "Carrefour", "Lidl", "Aldi",
    "Coop", "Mediaworld", "Unieuro", "Booking", "Airbnb",
    "RyanAir", "Easyjet", "PayPal", "Satispay", "Bancomat",
]

DEFAULT_OUT = Path("data") / "transactions.csv"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(
    n_transactions: int = N_TRANSACTIONS,
    n_clients: int = N_CLIENTS,
    out_path: Path | str = DEFAULT_OUT,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)

    # Distribute transactions across clients with a mild Zipf-like skew
    # so a few clients are heavier than others — better stress test for
    # variable-length sequences.
    weights = rng.dirichlet(np.ones(n_clients) * 1.5)
    counts = rng.multinomial(n_transactions, weights)
    counts = np.clip(counts, 5, None)  # ensure every client has enough txs
    # Re-balance to exactly n_transactions
    while counts.sum() > n_transactions:
        counts[counts.argmax()] -= 1
    while counts.sum() < n_transactions:
        counts[counts.argmin()] += 1

    rows = []
    for client_id, n in enumerate(counts):
        rows.extend(_generate_client(client_id, int(n), rng))

    df = pd.DataFrame(rows)
    df = df.sort_values(["client_id", "timestamp"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows × {len(df.columns)} cols → {out_path}")
    print(f"  clients: {df['client_id'].nunique()} "
          f"(min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f} tx/client)")
    return out_path


def _generate_client(client_id: int, n_tx: int, rng: np.random.Generator) -> list[dict]:
    """Generate ``n_tx`` transactions for one client with a coherent timeline."""
    # Per-client preferences — gives each client a stable footprint across rows.
    fav_merchants = rng.choice(MERCHANT_POOL, size=rng.integers(3, 8), replace=False)
    fav_mcc = rng.integers(*MCC_RANGE, size=rng.integers(2, 6))
    base_balance = float(rng.uniform(500, 10_000))

    # Sorted timestamps on a Poisson-like cadence
    start = TS_BASE + int(rng.integers(0, TS_RANGE // 2))
    gaps = rng.exponential(scale=86_400 * 2, size=n_tx).astype(np.int64)  # ~2-day mean
    timestamps = start + np.cumsum(gaps)

    rows = []
    balance = base_balance
    for ts in timestamps:
        # ~75% spese (negative), ~25% accrediti (positive)
        sign = -1 if rng.random() < 0.75 else 1
        amount = sign * float(rng.lognormal(mean=3.2, sigma=1.0))
        balance = max(0.0, balance + amount)

        rows.append({
            "client_id":  client_id,
            "timestamp":  int(ts),
            "importo":    round(amount, 2),
            "saldo_post": round(balance, 2),
            "merchant":   str(rng.choice(fav_merchants))
                          if rng.random() < 0.7
                          else str(rng.choice(MERCHANT_POOL)),
            "mcc":        int(rng.choice(fav_mcc)) if rng.random() < 0.6
                          else int(rng.integers(*MCC_RANGE)),
            "canale":     int(rng.integers(*CANALE_RANGE)),
            "macro_tipo": int(rng.integers(*MACRO_TIPO_RANGE)),
            "sotto_tipo": int(rng.integers(*SOTTO_TIPO_RANGE)),
            "divisa":     int(rng.integers(*DIVISA_RANGE)),
        })
    return rows


if __name__ == "__main__":
    generate()
