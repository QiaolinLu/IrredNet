# Training

[Back to the project](../README.md) · [Data format](data.md) · [Evaluation](inference.md)

## Standard configuration

Each run trains one tensor property. The standard configuration uses four
interaction layers, a 5 Å graph cutoff, and a variable-length coefficient head.

| Option | Default |
|---|---|
| `--epochs` | `200` |
| `--batch-size` | `32` |
| `--learning-rate` | `1e-3` |
| `--weight-decay` | `1e-5` |
| `--loss-space` | `coefficient` |
| `--loss` | `mse` |
| `--lr-schedule` | `none` |
| `--seed` | `32` |
| `--cutoff` | `5.0` |
| `--symprec` | `1e-5` |

The device defaults to CUDA when available, otherwise CPU. Set `--device cpu`
or `--device cuda:0` explicitly when needed.

```bash
python -m irrednet.train --target dielectric \
  --dataset /path/to/dielectric.pkl --epochs 200 --output-dir runs/dielectric

python -m irrednet.train --target piezoelectric \
  --dataset /path/to/piezoelectric.pkl --epochs 200 --output-dir runs/piezoelectric

python -m irrednet.train --target elastic \
  --dataset /path/to/elastic.pkl --epochs 200 --output-dir runs/elastic
```

Add `--split-file /path/to/split.json` for a fixed material-ID partition.
The resolved configuration and actual sample order are saved with each run.

## Independent dielectric scalar head

The dielectric model can learn its `0e` amplitude through an independent scalar
branch, while retaining equivariant feature–basis contractions for the `2e`
component. It continues to output only the symmetry-allowed coefficients.

```bash
python -m irrednet.train \
  --target dielectric \
  --dataset /path/to/dielectric.pkl \
  --split-file /path/to/dielectric_split.json \
  --epochs 200 --batch-size 32 --seed 32 \
  --loss-space tensor --loss huber \
  --lr-schedule linear --min-learning-rate 1e-5 \
  --dielectric-scalar-head independent \
  --evaluate-last --output-dir runs/dielectric-independent
```

The scalar branch is initialized using the training-set median and interquartile
range of the normalized scalar labels. Validation and test labels are not used
for this initialization. This option is defined only for dielectric tensors.

## Losses and optimization

`--loss-space coefficient` supervises the packed independent coefficients.
`--loss-space tensor` reconstructs the tensor through the fixed basis and
computes the loss in the original label shape. Both support `mse`, `huber`, and
`l1`. Tensor-space training keeps the original label's projection residual.

Coefficient MSE averages over all actual coefficients in a batch, so structures
with more degrees of freedom contribute more terms. No padded zero targets are
introduced. Elementwise Huber and L1 losses depend on the coordinate basis;
their coefficient-space and tensor-space forms are not interchangeable.

For a linear learning-rate schedule, use `--lr-schedule linear` and set
`--min-learning-rate`. The schedule advances after optimizer updates. A batch
containing only symmetry-forbidden tensors has no trainable coefficients and
skips optimization. If the entire training split has no coefficients, fitting
is unnecessary and the trainer stops with an explanation.

## Checkpoints and results

Outputs default to a new timestamped directory under `runs/`, or the path given
by `--output-dir`. Existing output directories are not overwritten.

| File | Contents |
|---|---|
| `config.json` | Training arguments, model settings, split and sample metadata |
| `history.jsonl` | Per-epoch training loss, learning rate and validation metrics |
| `best.pt` | Checkpoint selected by validation coefficient MAE |
| `last.pt` | Final model, optimizer, scheduler and random-number states |
| `metrics.json` | Validation-selected checkpoint's test metrics and history |
| `test_predictions.npz` | Validation-selected checkpoint's test predictions |
| `final_epoch_validation_metrics.json` | Final-epoch validation metrics |

`--evaluate-last` additionally saves `final_epoch_metrics.json` and
`test_final_epoch_predictions.npz`. These correspond to the final epoch, whereas
the usual test outputs correspond to `best.pt`.

Checkpoint selection remains **validation coefficient MAE**, including when
training with a tensor-space loss. If validation has no independent coefficients,
the training loss is used and recorded as the selection criterion.

Use `--skip-test` for training and validation only, then evaluate a chosen
checkpoint separately. It cannot be combined with `--evaluate-last`.

`last.pt` contains continuation state, but a `--resume` CLI is not currently
implemented.

## Short workflow checks

Check an external dataset with a small network:

```bash
python -m irrednet.train --target elastic \
  --dataset /path/to/elastic.pkl \
  --small --limit 12 --epochs 2 --batch-size 4 --device cpu \
  --output-dir runs/elastic-smoke
```

Remove `--small` and `--limit` for a full experiment. To check all three properties
without any material data, run `python scripts/smoke_test.py`.
