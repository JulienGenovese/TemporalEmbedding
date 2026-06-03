# Analisi bug — embeddingclient

Analisi statica di tutto `src/` + smoke test eseguiti (`src.model`, `src.encoder`,
pipeline dati e 5 step di training reali su `data/transactions.csv`).
Data: 2026-06-02.

I problemi sono ordinati per impatto. Per ognuno: descrizione, file:riga, effetto.

---

## Bug verificati a runtime

### 1. `MTMHead.full_recon` è codice morto (gradiente = `None`)

- **Dove:** `src/loss.py:124` (costruzione `self.full_recon`), `src/loss.py:138`
  (calcolo `preds["full_recon"]`), `src/loss.py:162,175` (`mtm_loss` itera solo
  le chiavi `cat_*` e `num_*`).
- **Problema:** `full_recon` viene costruita e calcolata a ogni forward
  (~16K parametri + una matmul `(B,T,d_model)→d_model` per batch), ma non entra
  mai nella loss. Nessuna chiave `full_recon` viene letta da `mtm_loss`, e non
  esiste un target corrispondente.
- **Verifica:** dopo 5 step di training reali,
  `model.backbone.mtm_head.full_recon.weight.grad is None` → la testa non si
  allena mai.
- **Effetto:** parametri inutili salvati nel checkpoint + compute sprecato. Il
  termine di loss documentato in `CLAUDE.md` e nel docstring di `MTMHead`
  ("Full transaction masking → MSE") **non esiste**: o manca l'implementazione,
  o la testa va rimossa.

### 2. La temperatura learnable è iniettata solo manualmente dal Trainer

- **Dove:** `src/loss.py:40-48` (`ContrastiveHead.forward` ritorna solo `z`),
  `src/loss.py:257` (`output.get("temperature", torch.tensor(0.07))`),
  `src/model.py:104-111` (il `forward` del modello NON aggiunge `"temperature"`),
  `src/train.py:173` e `src/train.py:257` (il Trainer la patcha a mano).
- **Problema:** `combined_pretrain_loss` legge la temperatura dall'output del
  modello, ma il modello non la espone mai. Funziona solo perché
  `Trainer._step`/`_eval_epoch` la inseriscono manualmente in `output`.
- **Effetto:** chiunque usi `PretrainLoss`/`combined_pretrain_loss` fuori dal
  `Trainer` ottiene **silenziosamente** temperatura fissa `0.07`; la temperatura
  learnable non riceve gradiente e diverge dal valore loggato. La temperatura
  dovrebbe essere restituita da `model.forward()`, non patchata dall'esterno.

---

## Incongruenze di correttezza (non crash, ma effetti reali)

### 3. MTM gira sull'output del FieldTransformer, non del SequenceTransformer

- **Dove:** `src/model.py:108` (`self.mtm_head(transaction_embeddings)`, dove
  `transaction_embeddings` è l'uscita del FieldTransformer, riga `model.py:95`);
  docstring fuorviante in `src/loss.py:129` ("sequence transformer outputs");
  diagramma in `CLAUDE.md`.
- **Problema:** l'MTM riceve gli embedding **pre-sequenza**. Il
  `SequenceTransformer` restituisce solo `h_cls` (`sequence_encoder.py:127-128`),
  quindi non può alimentare un MTM per-posizione.
- **Effetto:** l'MTM ricostruisce i campi mascherati **senza alcun contesto
  temporale / di sequenza** — è puramente intra-transazione. O è intenzionale
  (e il docstring/diagramma sono sbagliati) o è un errore di cablaggio che
  indebolisce molto il task MTM. Da chiarire come scelta di design.

### 4. Il target MTM numerico di `importo` perde il segno

- **Dove:** `src/train.py:148` (`targets[name] = feat.normalizer(batch[name])`),
  `src/encoder.py:91-92` (`NumericNormalizer.__call__` fa `x.abs()` per i signed).
- **Problema:** `importo` è `signed=True`; il target di regressione è il valore
  normalizzato in **valore assoluto**. La testa numerica predice solo la
  magnitudine, mai il segno; lo slot di segno (debito/credito) non è mai un
  target MTM.
- **Effetto:** il dataset genera ~15% di accrediti e rimborsi
  (`src/make_dataset.py:322`, `_generate_refund`), quindi è segnale informativo
  che l'MTM ignora del tutto.

---

## Secondario (scelta di scaling discutibile, non un bug netto)

### 5. `TimeAwarePositionalEncoding` usa `delta_t` grezzo in secondi

- **Dove:** `src/sequence_encoder.py:19-40` (frequenze fisse da 1 a
  `1/max_timescale = 1e-6`); `delta_t` arriva in secondi da
  `src/data.py:150-152`.
- **Problema:** la frequenza più alta ha periodo ~6 secondi, ma `delta_t` reali
  arrivano a giorni (10^5–10^6 s). I canali ad alta frequenza producono
  `sin/cos` completamente aliasati → rumore su circa metà delle dimensioni del PE.
- **Effetto:** degrada l'encoding temporale. Tipicamente si normalizza `delta_t`
  (es. in giorni/ore) o si alza `max_timescale`.

---

## Controllato e NON è un bug (per evitare falsi positivi)

- **Masking MTM:** azzera correttamente l'input prima del forward
  (`src/train.py:156`), nessun leakage — confermato a runtime.
- **Bucket disgiunti delle finestre:** garantiti dal guard `min_transactions`
  del sampler (`src/data.py:237`), nessuna collisione di slot
  (`R >= windows_per_client` ⇒ `slot % k == slot`).
- **Decomposizione timestamp:** `dow` corretto (2020-01-01 → mer=3,
  2021-01-01 → ven=5), verificato in `src/encoder.py` smoke test.
- **Ordine AMP:** `scale → backward → unscale_ → clip → step → update` corretto
  (`src/train.py:177-183`).
- **Overflow int64 nel double-hash di `merchant`** (`src/encoder.py:326-327`):
  si wrappa in modo deterministico, resta un hash valido in `[1, n_buckets-1]`.
- **Mismatch `__len__`/`__iter__` del sampler** (`src/data.py:253-269`):
  innocuo con i numeri di default (3200/8 e 800/8 esatti, nessun resto).
