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

## 1. Generare i dati sintetici

I comandi da usare sono:

```bash
uv run py generate --type vanilla
uv run py generate --type coherent
```

Genera un CSV sintetico (di default ~400.000 transazioni su 4.000 clienti).

### Modificare noise e altri parametri del generatore

I parametri del dato sintetico stanno in `src/datasets/utils/config.py`.
I default di `SamplingConfig` sono letti da `config.toml`, sezione `[dataset.sampling]`.
Il `noise_level` è letto da `[dataset.sampling]`, mentre i parametri specifici del dataset `coherent` sono letti da `[syntentic.coherent]`:

- `NoiseConfig.noise_level` (range `[0, 1]`)
- `AmountConfig.merchant_amount_weight` (range `[0, 1]`, usato in `coherent`)

I path di output sono separati per esperimento e definiti con:
- `folder`
- `name_file`
- `ext` (`csv`/`.csv` oppure `parquet`/`.parquet`; altri valori generano errore)

nelle sezioni:
- `[syntentic.vanilla]`
- `[syntentic.coherent]`

Il filename finale include automaticamente il noise level (es. `transactions_coherent_noise_0_90.csv`), e il file salvato (CSV o parquet) include anche la colonna `noise_level`.

Il parametro principale del rumore è:

- `NoiseConfig.noise_level` (default `0.9`, range `[0, 1]`)

Con questo valore vengono ricalcolati automaticamente:

- `p_offpattern`
- `p_global_merchant`
- `p_refund`
- `sigma_spending`

Altri parametri utili da modificare nello stesso file:

| Sezione | Parametri principali | Effetto |
|---|---|---|
| `SamplingConfig` | `n_transactions`, `n_clients`, `alpha_dirichlet`, `min_tx_per_client`, `seed`, `noise_level` (in `config.toml` → `[dataset.sampling]`) | volume dataset, distribuzione transazioni per cliente e livello di rumore globale |
| `AmountConfig` | `spending_probability`, `lognormal_sigma`, `merchant_amount_weight` (in `config.toml` → `[syntentic.coherent]`) | frequenza addebiti/accrediti, dispersione importi e bilanciamento client/merchant nel dataset `coherent` |
| `MerchantConfig` | `common_merchants`, `p_common_merchant`, pool merchant | varietà e pattern merchant |
| `CategoricalConfig` | `cocau_vocab`, `p_noise` | cardinalità categorie e rumore categorico |
| `OutputConfig` | `folder`, `name_file`, `ext` (in `config.toml` → `[syntentic.vanilla]`, `[syntentic.coherent]`) | path file (CSV/parquet) per esperimento con suffix automatico `noise_level` |

`--type` (oppure `-type`) supporta solo `vanilla` o `coherent`.

## 2. Addestrare il modello

Avvia il pre-addestramento end-to-end:

```bash
uv run py train
uv run py train --type hier
uv run py train --type base
```

`--type` (oppure `-type`) ha default `hier`.  
`base` è supportato dalla CLI e al momento usa la stessa pipeline di training di `hier`.

Con le impostazioni di default l'addestramento dura pochi minuti su CPU. Al termine
trovi sotto `checkpoints/`:

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
uv run py pred --type base
```

Il comando usa il checkpoint `checkpoints/model_final.pt`, legge il dataset da
`[model.hierTransformer.paths].train_path` e salva l'output in
`[model.hierTransformer.paths].pred_path/[model.hierTransformer.paths].pred_file_name`.

### Modificare parametri training e modello

I parametri stanno in `src/models/hier_transformer/hier_config.py`, con default letti da `config.toml`:

- `TrainingConfig`: parametri di training/dataloader
- `ModelConfig`: parametri architettura Transformer

Sezioni TOML del modello:

- `[model.hierTransformer.paths]`
- `[model.hierTransformer.training]`
- `[model.hierTransformer.architecture]`

Parametri training più usati (`TrainingConfig`):

| Parametro | Default | Significato |
|---|---|---|
| `epochs` | 30 | numero epoche |
| `seq_len` | 32 | lunghezza finestra temporale |
| `clients_per_batch` | 8 | clienti distinti per batch |
| `windows_per_pair` | 2 | finestre per coppia InfoNCE |
| `mask_prob` | 0.15 | quota di feature mascherate (MTM) |
| `contrastive_weight` | 0.5 | peso parte contrastiva della loss |
| `lr` | 3e-4 | learning rate |
| `weight_decay` | 0.01 | regolarizzazione AdamW |
| `lr_gamma` | 0.95 | decay esponenziale del LR per epoca |
| `val_frac` | 0.2 | quota clienti in validazione |
| `device` | `None` | auto-select (`cuda` se disponibile, altrimenti `cpu`) |

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

## 3. Visualizzare i risultati dell'addestramento

Esporta la storia del training su TensorBoard:

```bash
uv run py plot
uv run py plot --type hier
uv run py plot --type base --history checkpoints/history.json --runs-dir runs/base
uv run tensorboard --logdir runs
```

`plot` è un comando dedicato della CLI (non un'opzione di `train`).

## Verifica rapida (facoltativa)

Per controllare al volo che il modello si costruisca correttamente, senza CSV e senza
addestramento, gira un forward-pass su un batch sintetico in memoria:

```bash
uv run python -m src.models.hier_transformer.model
```

---

Per i dettagli sull'architettura e sul design schema-driven dell'encoder, vedi
`copilot.md`.
