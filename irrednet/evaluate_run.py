"""Evaluate a recorded checkpoint after validation-only configuration selection."""

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .data import SpaceGroupDataset, load_samples
from .model import load_coefficient_model
from .predict import save_predictions
from .splits import load_material_split
from .train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing config.json and checkpoints")
    parser.add_argument("--checkpoint", choices=("last.pt", "best.pt"), default="last.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=Path, help="External dataset path, overriding the recorded location")
    parser.add_argument("--split-file", type=Path, help="Same material-ID split at a new location")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    if args.dataset is not None:
        config["dataset"] = str(args.dataset.resolve())
    if args.split_file is not None:
        if not config.get("split_file"):
            parser.error("The original run did not use an explicit split; cannot replace its partition")
        config["split_file"] = str(args.split_file.resolve())
    samples = load_samples(config["dataset"])
    if config.get("split_file"):
        splits, provenance = load_material_split(config["split_file"], samples, dataset_path=config["dataset"])
        for name, indices in splits.items():
            expected = [config["sample_ids"][i] for i in config["splits"][name]]
            ids = [str(next(samples[int(i)][key] for key in ("JARVIS_ID", "jid", "id") if key in samples[int(i)]))
                   for i in indices]
            if ids != expected:
                raise ValueError("Evaluation material IDs differ from the recorded training split")
    else:
        if config.get("limit") is not None:
            samples = samples[:config["limit"]]
        splits, provenance = config["splits"], None
    selected = [samples[int(i)] for i in splits[args.split]]
    dataset = SpaceGroupDataset(selected, config["target"], symprec=config["symprec"], cutoff=config["cutoff"])
    expected_ids = [config["sample_ids"][i] for i in config["splits"][args.split]]
    if [graph.sample_id for graph in dataset.graphs] != expected_ids:
        raise ValueError("Evaluation material IDs/order differ from the recorded training split")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        collate_fn=SpaceGroupDataset.collate, num_workers=0)
    model, checkpoint = load_coefficient_model(args.run_dir / args.checkpoint, args.device)
    metrics = evaluate(model, loader, args.device, config["target"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {key: checkpoint[key] for key in ("version", "basis_version", "label_convention", "epoch")}
    result.update(checkpoint=str((args.run_dir / args.checkpoint).resolve()), split_provenance=provenance,
                  selection="fixed final epoch" if args.checkpoint == "last.pt" else checkpoint["selection_metric"])
    result[args.split] = metrics
    prediction_path = args.output_dir / (args.split + "_predictions.npz")
    predictions = save_predictions(model, loader, args.device, prediction_path)
    if not np.array_equal(predictions["sample_id"], expected_ids):
        raise ValueError("Exported predictions do not follow the recorded evaluation order")
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
