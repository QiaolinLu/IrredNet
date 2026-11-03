"""Exercise training, prediction and relocated evaluation without material data.

Run after installing this package: python scripts/smoke_test.py
All synthetic input files, checkpoints and predictions live in a temporary
directory and are removed on exit. These are workflow checks, not benchmarks.
"""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np
from pymatgen.core import Lattice, Structure
import torch

from irrednet.coefficients import build_coefficient_basis
from irrednet.data import match_label_shape
from irrednet.splits import SPLIT_VERSION, file_sha256


def run(module, *arguments, cwd):
    command = [sys.executable, "-m", "irrednet." + module, *map(str, arguments)]
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1",
                                 "PYTHONDONTWRITEBYTECODE": "1"}, timeout=180)
    if result.returncode:
        raise RuntimeError(f"{module} failed:\n{result.stdout}\n{result.stderr}")
    return result


def synthetic_samples(target):
    rng = np.random.default_rng(19)
    shape = {"dielectric": (3, 3), "piezoelectric": (3, 6), "elastic": (6, 6)}[target]
    samples = []
    for index in range(12):
        if index % 3 == 0:
            structure = Structure(Lattice.cubic(4.0 + .03 * index), [14], [[0, 0, 0]])
        else:
            structure = Structure(
                [[3.1 + .03 * index, 0, 0], [.2, 4.2, 0], [.4, .3, 5.3]],
                [6, 8, 14], [[.12, .23, .34], [.35, .19, .81], [.51, .41, .24]],
            )
        basis = build_coefficient_basis(structure, target)
        tensor = torch.tensor(basis.decode(rng.normal(size=basis.dim)))
        labels = match_label_shape(tensor[None], torch.empty((1, *shape)), target)[0].numpy()
        samples.append({"id": f"synthetic-{index}", "structure": structure, target: labels})
    return samples


def check_target(folder, target):
    folder.mkdir()
    samples = synthetic_samples(target)
    data_path = folder / "synthetic.pkl"
    data_path.write_bytes(pickle.dumps(samples))
    split_path = folder / "split.json"
    ids = [sample["id"] for sample in samples]
    split_path.write_text(json.dumps({
        "version": SPLIT_VERSION, "dataset_ids": ids,
        "dataset_sha256": file_sha256(data_path),
        "splits": {"train": ids[:8], "val": ids[8:10], "test": ids[10:]},
        "unused_ids": [],
    }))
    output = folder / "run"
    options = []
    if target == "dielectric":
        options = ["--loss-space", "tensor", "--loss", "huber", "--lr-schedule", "linear",
                   "--dielectric-scalar-head", "independent"]
    run("train", "--target", target, "--dataset", data_path, "--split-file", split_path,
        "--epochs", 2, "--batch-size", 4, "--small", "--device", "cpu", "--evaluate-last",
        "--output-dir", output, *options, cwd=folder)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2]
    assert all(np.isfinite(row["train_loss"]) for row in history)
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 2 and checkpoint["optimizer_state_dict"]["state"]

    structures_path = folder / "unlabeled.pkl"
    structures_path.write_bytes(pickle.dumps([
        {"id": sample["id"], "structure": sample["structure"]} for sample in samples
    ]))
    prediction_path = folder / "inference.npz"
    run("predict", "--checkpoint", output / "last.pt", "--dataset", structures_path,
        "--output", prediction_path, "--device", "cpu", cwd=folder)
    with np.load(prediction_path, allow_pickle=False) as result:
        assert "coefficient_labels" not in result.files
        assert result["sample_id"].tolist() == ids
        for index, (start, stop) in enumerate(zip(result["coefficient_ptr"][:-1],
                                                 result["coefficient_ptr"][1:])):
            tensor = result["coefficients"][start:stop] @ result["cartesian_basis"][start:stop]
            np.testing.assert_allclose(tensor, result["cartesian_tensor"][index].ravel(),
                                       atol=2e-6, rtol=1e-5)
        if target == "piezoelectric":
            zero = result["num_coefficients"] == 0
            assert zero.any()
            np.testing.assert_array_equal(result["cartesian_tensor"][zero], 0)

    moved = folder / "relocated"
    moved.mkdir()
    moved_data = data_path.rename(moved / data_path.name)
    moved_split = split_path.rename(moved / split_path.name)
    evaluation = folder / "evaluation"
    run("evaluate_run", "--run-dir", output, "--checkpoint", "last.pt", "--split", "test",
        "--dataset", moved_data, "--split-file", moved_split, "--device", "cpu",
        "--output-dir", evaluation, cwd=folder)
    recorded = json.loads((output / "final_epoch_metrics.json").read_text())["test"]
    actual = json.loads((evaluation / "metrics.json").read_text())["test"]
    assert recorded == actual
    with np.load(output / "test_final_epoch_predictions.npz") as original, \
            np.load(evaluation / "test_predictions.npz") as repeated:
        np.testing.assert_array_equal(original["coefficients"], repeated["coefficients"])
        np.testing.assert_array_equal(original["sample_id"], repeated["sample_id"])
    print(f"PASS {target}: 2 training epochs, unlabeled prediction, relocated evaluation", flush=True)


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="irrednet-smoke-") as directory:
        for target in ("dielectric", "piezoelectric", "elastic"):
            check_target(Path(directory) / target, target)
    print("All workflow checks passed; temporary inputs and outputs removed.")


if __name__ == "__main__":
    main()
