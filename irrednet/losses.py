"""Losses for a model whose only learned outputs are independent coefficients."""

from .coefficients import coefficient_loss
from .data import match_label_shape


def training_loss(model, coefficients, batch, target, *, space="coefficient", kind="mse"):
    """Return the mean loss and its actual number of scalar reduction terms.

    Tensor supervision differentiates through the fixed, per-structure basis;
    it does not introduce a learned tensor output head. Unlike MSE, elementwise
    Huber is not invariant to an orthogonal change of output coordinates.
    """
    if space == "coefficient":
        return coefficient_loss(coefficients, batch.coefficient_labels, kind), coefficients.numel()
    if space != "tensor":
        raise ValueError(f"Unknown loss space {space!r}")
    prediction = match_label_shape(model.reconstruct(coefficients, batch), batch.labels, target)
    return coefficient_loss(prediction.reshape(-1), batch.labels.reshape(-1), kind), prediction.numel()


def dielectric_scalar_statistics(dataset, train_indices):
    """Robust location/scale of the normalized 0e coefficient, from training only."""
    import math
    import numpy as np

    values = np.asarray([float(dataset[int(i)].labels.reshape(3, 3).trace()) / math.sqrt(3)
                         for i in train_indices], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite dielectric training labels are required for scalar initialization")
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    return {"shift": float(median), "scale": max(float(q75 - q25), 1e-6),
            "num_training_materials": len(values), "method": "training 0e median and interquartile range"}
