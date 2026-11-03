"""Predict packed independent coefficients and export their reconstruction bases."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .coefficients import BASIS_VERSION, MODEL_VERSION, LABEL_CONVENTION
from .data import SpaceGroupDataset, load_samples
from .model import load_coefficient_model


@torch.no_grad()
def save_predictions(model, loader, device, output):
    model.eval()
    arrays = {name: [] for name in (
        "coefficients", "cartesian_tensor", "canonical_basis", "cartesian_basis",
        "coefficient_frame", "space_group_number", "num_coefficients", "sample_id",
    )}
    coefficient_labels = []
    for batch in loader:
        batch = batch.to(device)
        coefficients = model(batch)
        arrays["coefficients"].append(coefficients.cpu().numpy())
        arrays["cartesian_tensor"].append(model.reconstruct(coefficients, batch).cpu().numpy())
        for name in ("canonical_basis", "cartesian_basis", "coefficient_frame", "space_group_number", "num_coefficients"):
            arrays[name].append(batch[name].cpu().numpy())
        arrays["sample_id"].append(np.asarray(batch.sample_id, dtype=str))
        if "coefficient_labels" in batch:
            coefficient_labels.append(batch.coefficient_labels.cpu().numpy())
    if not arrays["coefficients"]:
        raise ValueError("No structures to predict.")
    result = {name: np.concatenate(chunks, axis=0) for name, chunks in arrays.items()}
    result["coefficient_ptr"] = np.concatenate(([0], np.cumsum(result["num_coefficients"])))
    result["basis_version"] = np.asarray(BASIS_VERSION)
    result["model_version"] = np.asarray(MODEL_VERSION)
    result["label_convention"] = np.asarray(LABEL_CONVENTION)
    result["target"] = np.asarray(model.target)
    if coefficient_labels:
        result["coefficient_labels"] = np.concatenate(coefficient_labels)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Project pickle; labels are not required or read")
    source.add_argument("--structure", help="CIF/POSCAR file")
    parser.add_argument("--output", required=True, type=Path, help="New .npz file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if args.output.suffix != ".npz":
        parser.error("output must end in .npz")
    model, checkpoint = load_coefficient_model(args.checkpoint, args.device)
    if args.structure:
        from pymatgen.core import Structure
        samples = [{"structure": Structure.from_file(args.structure), "id": args.structure}]
    else:
        samples = load_samples(args.dataset)
    if args.limit is not None:
        samples = samples[:args.limit]
    dataset = SpaceGroupDataset(samples, model.target, symprec=checkpoint["symprec"],
                                cutoff=checkpoint["model_args"]["max_radius"], require_labels=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate)
    result = save_predictions(model, loader, args.device, args.output)
    print(f"Saved {len(dataset)} structures, {len(result['coefficients'])} independent coefficients to {args.output}")


if __name__ == "__main__":
    main()
