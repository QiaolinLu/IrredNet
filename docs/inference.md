# Inference and evaluation

[Back to the project](../README.md) · [Data format](data.md) · [Training](training.md)

## Predict from a structure

```bash
python -m irrednet.predict \
  --checkpoint /path/to/run/best.pt \
  --structure /path/to/structure.cif \
  --device cpu --output predictions.npz
```

Structure files are read with pymatgen. Prediction uses the cutoff, symmetry
tolerance and model settings recorded in the checkpoint.

For a batch of structures, replace `--structure` with `--dataset`:

```bash
python -m irrednet.predict \
  --checkpoint /path/to/run/last.pt \
  --dataset /path/to/structures.pkl \
  --batch-size 32 --device cuda:0 --output batch_predictions.npz
```

Tensor labels are not required or used for prediction. Output files must have a
`.npz` extension and must not already exist.

## Read the predictions

| Array | Contents |
|---|---|
| `coefficients` | Packed independent coefficients for all structures |
| `coefficient_ptr` | Start/stop offsets into the packed coefficient array |
| `num_coefficients` | Allowed dimension of each structure |
| `canonical_basis` | Basis tensors in the canonical lattice frame, flattened |
| `cartesian_basis` | Basis tensors in the input Cartesian frame, flattened |
| `cartesian_tensor` | Reconstructed full Cartesian tensors |
| `coefficient_frame` | Orthogonal lattice frame used during encoding |
| `space_group_number` | Detected space-group numbers |
| `sample_id` | Material identifiers in input order |

The file also records the target and model/basis/label convention versions.
Labeled evaluation exports additionally include `coefficient_labels`.

```python
import numpy as np

with np.load("predictions.npz", allow_pickle=False) as prediction:
    i = 0
    start, stop = prediction["coefficient_ptr"][i:i + 2]
    coefficients = prediction["coefficients"][start:stop]
    basis = prediction["cartesian_basis"][start:stop]
    tensor = prediction["cartesian_tensor"][i]

    np.testing.assert_allclose(
        coefficients @ basis, tensor.ravel(), atol=1e-5, rtol=1e-5
    )
```

The coefficients are coordinates in an orthonormal allowed basis. They generally
differ from conventional tensor components such as `C11` or `e15`.

## Evaluate a recorded experiment

```bash
python -m irrednet.evaluate_run \
  --run-dir /path/to/run \
  --checkpoint last.pt --split test --device cuda:0 \
  --output-dir evaluation-final
```

`--run-dir` contains `config.json` and the checkpoint. Choose `best.pt` for the
validation-selected model or `last.pt` for the final-epoch model. The output
directory must be new.

If the data has moved, add `--dataset /new/path/data.pkl` and, for a run using an
explicit split, `--split-file /new/path/split.json`. These override file locations;
material IDs and their evaluation order must still match the recorded run.
For runs without an explicit split, preserve the original dataset ordering.

## Metrics

Evaluation reports coefficient and tensor MAE/MSE, mean norm error, and
`ewt25`, `ewt10`, and `ewt05`. For each material, the relative error is

```text
relative error = ||prediction − label|| / (||label|| + 1e-5)
```

EwT is the fraction of materials with relative error strictly below the selected
threshold. Values are stored as fractions between 0 and 1. Mean norm error is
the average of per-material Euclidean/Frobenius error norms.

Tensor metrics use the original label shape. The norm of a compressed `(3, 6)`
or `(6, 6)` label is not the same as the norm of its expanded Cartesian tensor.
Orthonormal coefficient norms correspond to full Cartesian norms within the
allowed subspace.

Piezoelectric evaluation reports zero-degree-of-freedom and nonzero-degree-of-
freedom subsets separately. Dielectric evaluation also separates the isotropic
scalar `trace(T)/3` from its traceless component. Label projection residuals
measure the part of a label outside the symmetry-allowed space.

## Decomposition reports

```bash
python -m irrednet.report --structure /path/to/structure.cif --target elastic
python -m irrednet.report --demo --target all
```

The report gives the O(3)-to-space-group branching and allowed dimension.
Irrep indices are local spgrep indices, rather than Bilbao/Mulliken symbols.
