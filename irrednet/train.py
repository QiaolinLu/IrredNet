"""Train independent space-group coefficients directly from existing tensor labels."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .coefficients import BASIS_VERSION, MODEL_VERSION, LABEL_CONVENTION
from .data import SpaceGroupDataset, load_samples, match_label_shape
from .decomposition import PROPERTY_FORMULAS
from .model import IrredNetSpaceGroup
from .splits import load_material_split
from .losses import training_loss, dielectric_scalar_statistics


def save_checkpoint_atomic(checkpoint, path):
    """Keep the previous complete checkpoint until the new file is ready."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def training_state(model_checkpoint, optimizer, scheduler, config, history, best_epoch, best_score):
    """End-of-epoch state, including optimizer and RNGs needed for continuation.

    Only tensors and builtin types are stored, so weights_only=True can read it.
    best.pt remains the independently selected inference checkpoint.
    """
    numpy_state = np.random.get_state()
    return {
        **model_checkpoint,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "training_config": config, "history": history,
        "best_epoch": best_epoch, "best_score": best_score,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4],
        },
    }


def make_model_args(target, node_dim, *, small=False, interaction_type="standard", cutoff=5.0):
    from types import SimpleNamespace
    return SimpleNamespace(
        target=target, formula=PROPERTY_FORMULAS[target], interaction_type=interaction_type,
        use_attention=False, num_layers=2 if small else 4,
        node_attrs_irreps=f"{node_dim}x0e", node_feats_irreps="4x0e" if small else "32x0e",
        hidden_irreps=("4x0e+2x1o+2x2e+2x2o+1x3o+1x4e" if small else
                       "32x0e+32x0o+16x1e+16x1o+4x2e+4x2o+4x3e+4x3o+4x4e"),
        max_radius=cutoff, num_of_basis=4 if small else 20, num_polynomial_cutoff=5,
        radial_type="bessel", distance_transform=None, lmax=4 if target == "elastic" else 3,
        activation_type="gate", norm_type="LayerNorm", radial_MLP="[8, 8]" if small else "[64, 64, 64]",
        coefficient_hidden_dim=32 if small else 128,
    )


def split_indices(counts, seed):
    """80/10/10 split, with learnable samples in each set when at least 3 exist."""
    counts = np.asarray(counts)
    if len(counts) < 3:
        raise ValueError("At least three samples are required.")
    ids = np.random.default_rng(seed).permutation(len(counts))
    n_val = n_test = max(1, int(0.1 * len(ids)))
    n_train = len(ids) - n_val - n_test
    splits = {"train": ids[:n_train].copy(), "val": ids[n_train:n_train + n_val].copy(),
              "test": ids[n_train + n_val:].copy()}
    recipients = list(splits) if np.count_nonzero(counts) >= 3 else ["train"]
    for name in recipients:
        if np.any(counts[splits[name]] > 0):
            continue
        donors = [key for key in splits if key != name and np.any(counts[splits[key]] > 0)]
        if not donors:
            continue
        donor = max(donors, key=lambda key: np.count_nonzero(counts[splits[key]]))
        index = np.flatnonzero(counts[splits[donor]] > 0)[0]
        splits[name][0], splits[donor][index] = splits[donor][index], splits[name][0]
    return splits


@torch.no_grad()
def evaluate(model, loader, device, target):
    model.eval()
    tensor_metrics = ("tensor", "tensor_nonzero_dof", "tensor_zero_dof") if target == "piezoelectric" else ("tensor",)
    if target == "dielectric":
        tensor_metrics += ("dielectric_scalar", "dielectric_deviatoric")
    totals = {name: [0.0, 0.0, 0] for name in ("coefficient", *tensor_metrics, "label_projection_residual")}
    relative_errors = {name: [] for name in ("coefficient", *tensor_metrics)}
    norm_errors = {name: [] for name in relative_errors}
    for batch in loader:
        batch = batch.to(device)
        coefficients = model(batch)
        prediction = match_label_shape(model.reconstruct(coefficients, batch), batch.labels, target)
        projected_labels = match_label_shape(
            model.reconstruct(batch.coefficient_labels, batch), batch.labels, target
        )
        errors = {"coefficient": coefficients - batch.coefficient_labels,
                  "tensor": prediction - batch.labels,
                  "label_projection_residual": projected_labels - batch.labels}
        tensor_labels = {"tensor": batch.labels}
        if target == "dielectric":
            scalar_label = batch.labels.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
            scalar_error = errors["tensor"].diagonal(dim1=-2, dim2=-1).sum(-1) / 3
            identity = torch.eye(3, device=prediction.device, dtype=prediction.dtype)
            errors["dielectric_scalar"] = scalar_error[:, None]
            tensor_labels["dielectric_scalar"] = scalar_label[:, None]
            errors["dielectric_deviatoric"] = errors["tensor"] - scalar_error[:, None, None] * identity
            tensor_labels["dielectric_deviatoric"] = batch.labels - scalar_label[:, None, None] * identity
        if target == "piezoelectric":
            for name, mask in (("tensor_nonzero_dof", batch.num_coefficients > 0),
                               ("tensor_zero_dof", batch.num_coefficients == 0)):
                errors[name] = errors["tensor"][mask]
                tensor_labels[name] = batch.labels[mask]
        for name, error in errors.items():
            totals[name][0] += error.abs().sum().item()
            totals[name][1] += error.square().sum().item()
            totals[name][2] += error.numel()
        for name in relative_errors:
            if name == "coefficient":
                squared_error = coefficients.new_zeros(batch.num_graphs).index_add(
                    0, batch.coefficient_batch, errors[name].square())
                squared_label = coefficients.new_zeros(batch.num_graphs).index_add(
                    0, batch.coefficient_batch, batch.coefficient_labels.square())
                valid = batch.num_coefficients > 0
                error_norm, label_norm = squared_error[valid].sqrt(), squared_label[valid].sqrt()
            else:
                error_norm = errors[name].flatten(1).norm(dim=1)
                label_norm = tensor_labels[name].flatten(1).norm(dim=1)
            norm_errors[name].extend(error_norm.cpu().tolist())
            relative_errors[name].extend((error_norm / (label_norm + 1e-5)).cpu().tolist())
    result = {name: {"mae": absolute / max(count, 1), "mse": squared / max(count, 1), "count": count}
              for name, (absolute, squared, count) in totals.items()}
    for name, values in relative_errors.items():
        values = np.asarray(values)
        result[name].update({"num_materials": len(values), "relative_epsilon": 1e-5,
                             "mean_norm_error": float(np.mean(norm_errors[name])) if len(values) else None})
        for key, threshold in (("ewt25", .25), ("ewt10", .10), ("ewt05", .05)):
            result[name][key] = float(np.mean(values < threshold)) if len(values) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=PROPERTY_FORMULAS)
    parser.add_argument("--dataset", required=True, help="Existing preprocessed tensor pickle (read only)")
    parser.add_argument("--split-file", type=Path, help="Exact train/val/test material-ID manifest")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", choices=("none", "linear"), default="none")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--loss", choices=("mse", "huber", "l1"), default="mse", help="Elementwise loss in --loss-space")
    parser.add_argument("--loss-space", choices=("coefficient", "tensor"), default="coefficient",
                        help="Supervise coefficients directly or their fixed-basis tensor reconstruction")
    parser.add_argument("--dielectric-scalar-head", choices=("contract", "independent"), default="contract")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interaction-type", choices=("standard",), default="standard")
    parser.add_argument("--small", action="store_true", help="Small backbone and coefficient decoder for smoke runs")
    parser.add_argument("--limit", type=int, help="First N samples, for smoke runs")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run")
    parser.add_argument("--evaluate-last", action="store_true",
                        help="Also export final-epoch test results, separately from validation-best results")
    parser.add_argument("--skip-test", action="store_true", help="Validate only; do not evaluate or export test predictions")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    if args.limit is not None and args.limit < 3:
        parser.error("limit must be at least 3")
    if args.split_file is not None and args.limit is not None:
        parser.error("An explicit split cannot be combined with --limit")
    if args.skip_test and args.evaluate_last:
        parser.error("--skip-test cannot be combined with --evaluate-last")
    if args.dielectric_scalar_head == "independent" and args.target != "dielectric":
        parser.error("--dielectric-scalar-head independent requires --target dielectric")
    if args.lr_schedule == "linear" and (
            not np.isfinite(args.min_learning_rate) or not 0 < args.min_learning_rate <= args.learning_rate):
        parser.error("min-learning-rate must be positive and no larger than learning-rate")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    path = args.dataset
    samples = load_samples(path)
    if args.limit is not None:
        samples = samples[:args.limit]
    if len(samples) < 3:
        parser.error("At least three samples are required")
    splits, split_provenance = None, None
    if args.split_file is not None:
        splits, split_provenance = load_material_split(args.split_file, samples, dataset_path=path)
        print(f"Verified explicit split: {', '.join(f'{name}={len(ids)}' for name, ids in splits.items())}; "
              f"unused={split_provenance['num_unused']}", flush=True)
    print(f"Preparing coefficient labels for {len(samples)} {args.target} structures...", flush=True)
    dataset = SpaceGroupDataset(samples, args.target, symprec=args.symprec, cutoff=args.cutoff,
                                progress_every=100)
    isolated_count = sum(graph.num_edges == 0 for graph in dataset.graphs)
    print(f"Prepared all structures; graphs without neighbors={isolated_count} "
          "(retained using on-site features).", flush=True)
    counts = [graph.num_coefficients for graph in dataset.graphs]
    if not any(counts):
        parser.error("All tensors are symmetry-forbidden (zero coefficients); no model fitting is needed")
    if splits is None:
        splits = split_indices(counts, args.seed)
    if not any(counts[i] for i in splits["train"]):
        parser.error("Training split contains no independent coefficients")
    loaders = {name: DataLoader(Subset(dataset, ids.tolist()), batch_size=args.batch_size,
                               shuffle=name == "train", collate_fn=SpaceGroupDataset.collate,
                               num_workers=0, drop_last=False)
               for name, ids in splits.items()}
    model_args = make_model_args(args.target, dataset.graphs[0].num_node_features,
                                 small=args.small, interaction_type=args.interaction_type, cutoff=args.cutoff)
    scalar_initialization = None
    model_args.dielectric_scalar_head = args.dielectric_scalar_head
    if args.dielectric_scalar_head == "independent":
        scalar_initialization = dielectric_scalar_statistics(dataset, splits["train"])
        model_args.dielectric_scalar_shift = scalar_initialization["shift"]
        model_args.dielectric_scalar_scale = scalar_initialization["scale"]
    device = torch.device(args.device)
    model = IrredNetSpaceGroup(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Step the optional schedule after actual optimization updates. Mixed batches
    # are shuffled, so use the maximum step count if some samples have no modes.
    total_steps = args.epochs * len(loaders["train"])
    scheduler = (torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args.min_learning_rate / args.learning_rate,
        total_iters=total_steps) if args.lr_schedule == "linear" else None)
    output = args.output_dir or Path("runs") / (
        f"{args.target}-coefficients-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    use_validation = any(counts[i] for i in splits["val"])
    selection_metric = "val_coefficient_mae" if use_validation else "train_loss"
    if not use_validation:
        print("Validation has no independent coefficients; selecting checkpoints by training coefficient loss.")
    config = {**vars(args), "dataset": str(Path(path).resolve()), "output_dir": str(output.resolve()),
              "split_file": str(args.split_file.resolve()) if args.split_file is not None else None,
              "split_provenance": split_provenance,
              "version": MODEL_VERSION, "basis_version": BASIS_VERSION, "label_convention": LABEL_CONVENTION,
              "model_args": vars(model_args), "splits": {k: v.tolist() for k, v in splits.items()},
              "scalar_initialization": scalar_initialization,
              "num_coefficients": counts, "selection_metric": selection_metric,
              "num_graphs_without_neighbors": isolated_count,
              "sample_ids": [graph.sample_id for graph in dataset.graphs],
              "space_groups": [graph.space_group_number for graph in dataset.graphs]}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training on {device}: "
          f"{', '.join(f'{name}={len(ids)}' for name, ids in splits.items())}; "
          f"parameters={sum(p.numel() for p in model.parameters())}; output={output.resolve()}",
          flush=True)
    best_score, best_epoch, history = float("inf"), None, []
    for epoch in range(args.epochs):
        epoch_started = time.monotonic()
        model.train()
        total_loss, total_coefficients, total_loss_terms = 0.0, 0, 0
        for step, batch in enumerate(loaders["train"], 1):
            if batch.coefficient_labels.numel() == 0:
                continue  # no fake zero targets, gradients or AdamW decay for this batch
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_terms = training_loss(model, model(batch), batch, args.target,
                                             space=args.loss_space, kind=args.loss)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * loss_terms
            total_loss_terms += loss_terms
            total_coefficients += batch.coefficient_labels.numel()
            if step == 1 or step % 50 == 0 or step == len(loaders["train"]):
                print(f"Epoch {epoch + 1}/{args.epochs} step {step}/{len(loaders['train'])}: "
                      f"{args.loss_space}_{args.loss}_loss={total_loss / total_loss_terms:.6g}; "
                      f"elapsed={time.monotonic() - epoch_started:.1f}s", flush=True)
        train_loss = total_loss / total_loss_terms
        val = evaluate(model, loaders["val"], device, args.target)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "loss_space": args.loss_space,
                  "loss_kind": args.loss, "train_loss_count": total_loss_terms,
                  "train_coefficient_count": total_coefficients,
                  "learning_rate": optimizer.param_groups[0]["lr"], "val": val}
        if args.loss_space == "coefficient":
            record["train_coefficient_loss"] = train_loss
        history.append(record)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = val["coefficient"]["mae"] if use_validation else train_loss
        if not np.isfinite(score):
            raise RuntimeError("Nonfinite checkpoint-selection metric.")
        model_checkpoint = {"model_state_dict": model.state_dict(), "model_args": vars(model_args),
                            "version": MODEL_VERSION, "basis_version": BASIS_VERSION,
                            "label_convention": LABEL_CONVENTION, "epoch": epoch + 1,
                            "selection_score": score,
                            "symprec": args.symprec, "selection_metric": selection_metric}
        if score < best_score:
            best_score, best_epoch = score, epoch + 1
            save_checkpoint_atomic(model_checkpoint, output / "best.pt")
        save_checkpoint_atomic(training_state(model_checkpoint, optimizer, scheduler, config,
                                               history, best_epoch, best_score), output / "last.pt")
    validation = {"version": MODEL_VERSION, "basis_version": BASIS_VERSION,
                  "label_convention": LABEL_CONVENTION, "epoch": args.epochs,
                  "checkpoint": "last.pt", "selection": "fixed final epoch", "val": history[-1]["val"]}
    (output / "final_epoch_validation_metrics.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if args.skip_test:
        (output / "metrics.json").write_text(json.dumps({
            "version": MODEL_VERSION, "basis_version": BASIS_VERSION, "label_convention": LABEL_CONVENTION,
            "best_epoch": best_epoch, "final_epoch": args.epochs, "history": history,
            "validation_best": history[best_epoch - 1]["val"], "validation_final_epoch": history[-1]["val"],
            "test_evaluated": False,
        }, indent=2), encoding="utf-8")
        print(f"Saved {output.resolve()}; validation only; test evaluation deferred", flush=True)
        return
    from .predict import save_predictions
    final_test = None
    if args.evaluate_last:
        final_test = evaluate(model, loaders["test"], device, args.target)
        save_predictions(model, loaders["test"], device, output / "test_final_epoch_predictions.npz")
        (output / "final_epoch_metrics.json").write_text(json.dumps({
            "version": MODEL_VERSION, "basis_version": BASIS_VERSION,
            "label_convention": LABEL_CONVENTION, "epoch": args.epochs,
            "checkpoint": "last.pt", "selection": "fixed final epoch, no test-set selection",
            "test": final_test,
        }, indent=2), encoding="utf-8")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {"version": MODEL_VERSION, "basis_version": BASIS_VERSION,
               "label_convention": LABEL_CONVENTION, "best_epoch": checkpoint["epoch"],
               "history": history, "test": evaluate(model, loaders["test"], device, args.target)}
    if final_test is not None:
        metrics.update({"final_epoch": args.epochs, "test_final_epoch": final_test})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_predictions(model, loaders["test"], device, output / "test_predictions.npz")
    print(f"Saved {output.resolve()}; test={metrics['test']}")


if __name__ == "__main__":
    main()
