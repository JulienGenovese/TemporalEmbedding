# embeddingclient

## Di cosa si tratta

Una banca conosce i suoi clienti soprattutto attraverso le **transazioni**: pagamenti,
prelievi, bonifici, accrediti. Presi singolarmente questi movimenti dicono poco; presi
insieme raccontano abitudini, capacità di spesa, stile di vita.

`embeddingclient` trasforma questa storia transazionale in un **embedding del cliente**:
un singolo vettore numerico che riassume il comportamento di una persona. Clienti con
abitudini simili ottengono vettori vicini tra loro, clienti diversi vettori lontani.

Questo vettore è una rappresentazione riutilizzabile: può servire per raggruppare clienti
simili, trovare i "vicini" di un cliente dato, oppure come input pronto per altri modelli
(scoring, raccomandazione, rilevamento anomalie) senza dover ripartire ogni volta dai dati
grezzi.

## Come funziona, in breve

Il cuore del progetto è un modello Transformer **gerarchico**, che legge i dati su due
livelli:

1. **la singola transazione** — combina i suoi campi (importo, saldo, esercente, categoria
   merceologica, canale, ecc.) in una rappresentazione compatta;
2. **la sequenza di transazioni** — mette in relazione i movimenti nel tempo, tenendo
   conto di *quanto* tempo passa tra l'uno e l'altro, e li condensa nell'embedding finale
   del cliente.

Il modello impara **da solo**, senza etichette, con due esercizi complementari:

- **ricostruzione** — alcuni campi delle transazioni vengono nascosti e il modello deve
  indovinarli, così è costretto a capire come è fatta una transazione "tipica";
- **confronto** — due spezzoni della storia dello stesso cliente devono produrre
  embedding simili, mentre quelli di clienti diversi devono restare distinti.

Il progetto include anche un **generatore di dati sintetici**, così l'intera pipeline può
essere provata end-to-end su una normale CPU senza bisogno di dati reali.

---

# Come usarlo

## Prerequisiti

- **Python 3.10–3.11** (come da `pyproject.toml`)
- [**uv**](https://docs.astral.sh/uv/) come gestore dell'ambiente e delle dipendenze

## Uso con Dev Container (PyTorch)

Il repository include una configurazione in `.devcontainer/` basata su immagine
**`pytorch/pytorch`**.

Apri il progetto in VS Code e scegli:

1. `Dev Containers: Reopen in Container`

Alla prima creazione del container vengono installate automaticamente le dipendenze
del progetto.

Installa tutto con:

```bash
uv sync
```

> È disponibile una CLI progetto (`py`) implementata con **Typer** e registrata in `pyproject.toml`.
> Usa i comandi come `uv run py <comando> ...`.
> Per vedere tutti i comandi: `uv run py --help`.

Per eseguire un esperimento end-to-end tra quelli disponibili (`simple_spatial`,
`simple_delta` o `simple_calendar`: generazione dati, training, predizione
embedding e analisi di perturbazione):

```bash
uv run python run_experiment.py simple_spatial
uv run python run_experiment.py simple_delta
uv run python run_experiment.py simple_calendar
```

## 1. Generare i dati sintetici

I comandi da usare sono:

```bash
uv run py synthetic --type simple_spatial
uv run py synthetic --type simple_delta
uv run py synthetic --type simple_calendar
```

Ogni esecuzione genera due split (`train` e `pred`) per il tipo scelto.

### Modificare i parametri del generatore

La configurazione runtime è gestita da `src/config.py` (singleton `config`), letta da `config.toml`.
Nei generatori sintetici i valori vengono presi direttamente con:

```python
from src.config import config
config.get("synthetic.simple_spatial", "output", value_type=Path)
```

Sezioni/chiavi principali in `config.toml`:

| Sezione | Parametri principali | Effetto |
|---|---|---|
| `[synthetic.simple_spatial]`, `[synthetic.simple_delta]`, `[synthetic.simple_calendar]` | `seed`, `output` | seed RNG e path output per esperimento (file distinti per split: `*_train`, `*_pred`) |
| `[synthetic.timing]` | `ts_base`, `ts_range`, `day` | finestra temporale della simulazione |

Le cardinalità/forme dei cluster sintetici (`n_clients`, distribuzioni importi, ecc.) sono definite nei file esperimento:
- `src/datasets/experiments/simple_spatial.py`
- `src/datasets/experiments/simple_delta.py`
- `src/datasets/experiments/simple_calendar.py`

`--type` (oppure `-t`) supporta `simple_spatial`, `simple_delta`, `simple_calendar`.

## 2. Addestrare il modello

Avvia il pre-addestramento end-to-end:

```bash
uv run py train
uv run py train --type hier
uv run py train -t hier
```

`--type` (oppure `-t`) ha default `hier`.

Con le impostazioni di default l'addestramento dura pochi minuti su CPU. Al termine
trovi gli artifact sotto `model_artifacts/<dataset>/<data-training>/`, con
`model_artifacts/latest/` aggiornato all'ultimo training:

| File                      | Contenuto                                                  |
|---------------------------|------------------------------------------------------------|
| `model_final.pt`          | i pesi del modello addestrato                              |
| `history.json`            | l'andamento dell'addestramento, registrato ad ogni passo   |
| `train_eval_history.json` | metriche di valutazione sul set di training, per epoca     |
| `val_history.json`        | metriche di valutazione sul set di validazione, per epoca  |

### Predire embedding per finestra

Dopo il training puoi generare gli embedding finestra-per-finestra:

```bash
uv run py pred
uv run py pred --type hier
uv run py pred -t hier
```

Il comando usa il modello `model_artifacts/latest/model_final.pt`, legge il dataset da
`[model.hier_transformer.paths].pred_input_path` e salva l'output in
`[model.hier_transformer.paths].pred_output_path` (path completo file, es. `data/pred/pred_embeddings.csv`).

### Analisi di perturbazione

Per misurare quanto ogni variabile influenzi gli embedding:

```bash
uv run py perturbation
uv run py perturbation --type hier
uv run py perturbation -t hier
uv run py perturbation --analysis classification
uv run py perturbation --analysis sensibility
```

Con `--analysis sensibility` il comando ricalcola gli embedding dopo aver
permutato le colonne configurate in `src/eval/sensibility.py` e salva il report
CSV in
`[model.hier_transformer.perturbation].output_path` (default:
`model_artifacts/perturbation.csv`).
Con `--analysis classification` addestra una regressione logistica sugli
embedding puliti per predire la label `cluster`, poi ripredice il cluster sugli
embedding ottenuti permutando una colonna e misura il calo di accuratezza/F1. Il
report viene salvato in una sottodirectory derivata dal dataset di input
`[model.hier_transformer.paths].pred_input_path`; per esempio, con il default
`classification_output_path = "model_artifacts/classification_perturbation.csv"` e input
`data/transactions_simple_spatial_pred.csv`, l'output diventa
`model_artifacts/simple_spatial/classification_perturbation.csv`.
Per limitare l'analisi a una sola colonna, imposta
`[model.hier_transformer.perturbation].column` in `config.toml` (per esempio `delta_t`).

### Modificare parametri training e modello

I parametri stanno in `src/models/hier_transformer/hier_config.py`, con default letti da `config.toml`:

- `HierTransformerConfig`: contenitore top-level del pipeline gerarchico
- `PathsConfig`: path di input/output e directory artifact
- `DataPipelineConfig`: finestratura e batching (`seq_len`, `pred_windows_per_client`, `clients_per_batch`, `train_windows_per_client`)
- `TrainingConfig`: ottimizzazione e validazione (`epochs`, `mask_prob`, `contrastive_weight`, `lr`, `weight_decay`, `lr_gamma`, ...)
- `RuntimeConfig`: device e seed
- `PerturbationConfig`: colonna, path e opzioni per le analisi di perturbazione
- `ModelConfig`: parametri architettura Transformer

Sezioni TOML del modello:

- `[model.hier_transformer.paths]`
- `[model.hier_transformer.data]`
- `[model.hier_transformer.training]`
- `[model.hier_transformer.runtime]`
- `[model.hier_transformer.architecture]`
- `[model.hier_transformer.perturbation]`

Parametri data pipeline più usati (`DataPipelineConfig`):

| Parametro | Default | Significato |
|---|---|---|
| `seq_len` | 32 | lunghezza finestra temporale |
| `pred_windows_per_client` | 4 | finestre deterministiche per cliente in prediction |
| `clients_per_batch` | 8 | clienti distinti per batch |
| `train_windows_per_client` | 2 | finestre generate per cliente in ogni batch InfoNCE |

Parametri training più usati (`TrainingConfig`):

| Parametro | Default | Significato |
|---|---|---|
| `epochs` | 30 | numero epoche |
| `mask_prob` | 0.15 | quota di feature mascherate (MTM) |
| `contrastive_weight` | 0.5 | peso parte contrastiva della loss |
| `lr` | 3e-4 | learning rate |
| `weight_decay` | 0.01 | regolarizzazione AdamW |
| `lr_gamma` | 0.95 | decay esponenziale del LR per epoca |
| `val_frac` | 0.2 | quota clienti in validazione |

Parametri runtime più usati (`RuntimeConfig`):

| Parametro | Default | Significato |
|---|---|---|
| `device` | `gpu` | usa CUDA (`cpu` per forzare CPU) |
| `seed` | 0 | seed condiviso per split/sampling |

Parametri modello (`ModelConfig`):

| Parametro | Default | Significato |
|---|---|---|
| `d_field` | 64 | embedding di ogni campo transazione |
| `d_model` | 128 | dimensione interna del modello |
| `field_n_layers` | 2 | layer attenzione intra-transazione |
| `field_n_heads` | 4 | teste attenzione intra-transazione |
| `seq_n_layers` | 4 | layer attenzione sulla sequenza |
| `seq_n_heads` | 8 | teste attenzione sulla sequenza |
| `dim_feedforward` | 512 | dimensione FFN dei blocchi Transformer |
| `dropout` | 0.1 | dropout globale |
| `n_frequencies` | 16 | frequenze sin/cos per encoder numerico |
| `time_alpha_init` | 0.1 | valore iniziale del gate appreso che scala il time-delta encoding |

## 3. Visualizzare i risultati dell'addestramento

Esporta la storia del training su TensorBoard:

```bash
uv run py plot
uv run py plot --type tensorboard
uv run py plot --type tensorboard --experiment simple_spatial
uv run py plot --type tensorboard --experiment simple_spatial/2026-06-19_17-25-00
uv run py plot --type tensorboard --serve
```

`plot` è un comando dedicato della CLI (non un'opzione di `train`) e al momento
supporta solo l'export `tensorboard`.
Senza `--experiment` esporta l'ultimo training (`latest`); con `--experiment`
puoi scegliere una sottocartella artifact specifica.
Usa `--serve` per lanciare direttamente la UI TensorBoard dopo l'export. Nel
selettore **Run** di TensorBoard trovi gli esperimenti esportati con i sub-run
`train`, `val` e `random`, così le metriche omonime sono sovrapposte negli
stessi grafici.
I path di input/output per `plot` sono configurati in `[model.plot]` su `config.toml`.

## Verifica rapida (facoltativa)

Per controllare al volo che il modello si costruisca correttamente, senza CSV e senza
addestramento, gira un forward-pass su un batch sintetico in memoria:

```bash
uv run python -m src.models.hier_transformer.model
```

---

Per i dettagli sull'architettura e sul design schema-driven dell'encoder, vedi
`.github/copilot-instructions.md`.
