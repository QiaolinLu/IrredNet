# Data preparation

[Back to the project](../README.md)

## Sample format

Training reads an external, trusted pickle file containing a list or tuple of
sample dictionaries. A sample has the following fields:

| Field | Contents |
|---|---|
| `id` | A unique material identifier; `JARVIS_ID` and `jid` are also accepted |
| `structure` | A `pymatgen.core.Structure` in the same frame as its tensor label |
| Property name | An array under `dielectric`, `piezoelectric`, or `elastic` |

For example, given an existing structure and label array:

```python
sample = {
    "id": "sample-001",
    "structure": structure,
    "elastic": elastic_tensor,
}
```

The loader also accepts `p_input.structure`, `corrected_structure`, and JARVIS
`atoms` dictionaries. `p_input.structure` takes precedence over the top-level
structure. See [`sample_structure`](../irrednet/data.py) for the full lookup order.

The atom encoder supports atomic numbers **1–101**. Structures must be ordered
and fully occupied. A single training run must use a consistent label shape.

## Shapes and units

| Property | Accepted shape | Unit convention |
|---|---|---|
| Dielectric | `(3, 3)` | Relative permittivity, dimensionless |
| Piezoelectric | `(3, 6)` or `(3, 3, 3)` | C/m², piezoelectric stress tensor |
| Elastic | `(6, 6)` or `(3, 3, 3, 3)` | GPa |

The compressed index order is:

```text
xx, yy, zz, xy, yz, xz
```

There are no engineering-shear or Mandel scaling factors. For an elastic matrix,
this order applies to both axes. For a piezoelectric matrix, it applies to the
six columns.

The loader does not convert physical units. Convert raw `elastic_total_kbar`
values to GPa by dividing by 10 before storing them under `elastic`. Values
already in GPa should be used as provided.

Rotate a structure and its labels together. A label must correspond to the input
Cartesian frame, even when the lattice is skewed or has been reoriented.

## Fixed train / validation / test splits

Use `--split-file` to supply an ordered partition by material ID:

```json
{
  "version": "material_id_split_v1",
  "dataset_ids": ["sample-a", "sample-b", "sample-c"],
  "splits": {
    "train": ["sample-a"],
    "val": ["sample-b"],
    "test": ["sample-c"]
  },
  "unused_ids": []
}
```

The identifiers above illustrate the schema. Actual splits are supplied with
the external data.

IDs must be unique, partitions must not overlap, and the three partitions plus
`unused_ids` must cover the entire dataset. Each training, validation, and test
partition must contain at least one sample. Explicit splits retain their listed
order and cannot be combined with `--limit`.

Without a split file, the trainer creates an approximately 80/10/10 partition
using `--seed`. When at least three structures have nonzero tensor degrees of
freedom, it ensures that every partition contains a learnable sample. A shared
seed alone does not establish identical splits between different pipelines.

## Symmetry processing

The dataset builds a periodic graph and an independent tensor basis for each
structure. `--cutoff` controls the graph radius, and `--symprec` controls symmetry
identification. Graphs without neighbors remain valid through the backbone's
on-site feature path.

Coefficient labels are obtained by projecting tensor labels onto the allowed
basis. The original labels remain available, and evaluation reports their
projection residual. A nonzero residual can reflect noise, tensor conventions,
or a mismatch between structure symmetry and the label.

For inference, the same sample format can be used with the property field
omitted. A single structure can also be read directly from CIF or POSCAR.
