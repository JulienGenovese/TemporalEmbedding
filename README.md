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

- **Python 3.14**
- [**uv**](https://docs.astral.sh/uv/) come gestore dell'ambiente e delle dipendenze

Installa tutto con:

```bash
uv sync
```

> Gli script vivono in `src/` e usano import relativi: vanno sempre lanciati come modulo
> (`python -m src.<nome>`), mai come `python src/<nome>.py`.

## 1. Generare i dati

Crea un dataset sintetico di transazioni (~10.000 righe, 100 clienti) in
`data/transactions.csv`:

```bash
uv run python -m src.make_dataset
```

È il punto di partenza: serve a far girare l'addestramento senza dati reali.

## 2. Addestrare il modello

Avvia il pre-addestramento end-to-end:

```bash
uv run python -m src.train
```

Con le impostazioni di default l'addestramento dura pochi minuti su CPU. Al termine
trovi sotto `checkpoints/`:

| File                      | Contenuto                                                  |
|---------------------------|------------------------------------------------------------|
| `model_final.pt`          | i pesi del modello addestrato                              |
| `history.json`            | l'andamento dell'addestramento, registrato ad ogni passo   |
| `train_eval_history.json` | metriche di valutazione sul set di training, per epoca     |
| `val_history.json`        | metriche di valutazione sul set di validazione, per epoca  |

### Cambiare le impostazioni

Lo script **non accetta argomenti da riga di comando**: tutti gli iper-parametri stanno
in `src/config.py`. Per modificarli (numero di epoche, learning rate, dimensione del
modello, ecc.) basta cambiare i valori in quel file. I più usati:

| Parametro            | Default | Significato                                       |
|----------------------|---------|---------------------------------------------------|
| `epochs`             | 20      | numero di epoche di addestramento                 |
| `seq_len`            | 32      | quante transazioni per finestra                   |
| `clients_per_batch`  | 8       | clienti distinti per batch                        |
| `mask_prob`          | 0.15    | quanti campi nascondere nell'esercizio di ricostruzione |
| `contrastive_weight` | 0.5     | peso dell'esercizio di confronto nella loss       |
| `lr`                 | 3e-4    | learning rate                                     |
| `val_frac`           | 0.2     | frazione di clienti tenuti per la validazione     |
| `device`             | `None`  | `None` = automatico (GPU se disponibile, altrimenti CPU) |

## 3. Visualizzare i risultati dell'addestramento

Genera i grafici a partire dalla storia salvata durante il training:

```bash
uv run python -m src.plots
```

Legge `checkpoints/history.json` e scrive in `checkpoints/plots/`:

- **`training_curves.png`** — riepilogo a 6 pannelli con l'andamento della loss,
  dell'accuratezza e di altre metriche;
- **`mtm_breakdown.png`** — l'andamento della ricostruzione, campo per campo.

Opzioni utili:

```bash
uv run python -m src.plots --smoothing 5        # media mobile più ampia sulle curve
uv run python -m src.plots --tensorboard        # esporta anche per TensorBoard
```

Con `--tensorboard` i dati finiscono sotto `runs/` e si esplorano con:

```bash
uv run tensorboard --logdir runs
```

## Verifica rapida (facoltativa)

Per controllare al volo che il modello si costruisca correttamente, senza CSV e senza
addestramento, gira un forward-pass su un batch sintetico in memoria:

```bash
uv run python -m src.model
```

---

Per i dettagli sull'architettura e sul design schema-driven dell'encoder, vedi
`CLAUDE.md`.
