import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from irrednet.train import save_checkpoint_atomic, training_state


class TestTrainingState(unittest.TestCase):
    def test_checkpoint_restores_the_next_optimizer_update_and_random_draws(self):
        torch.manual_seed(45)
        np.random.seed(45)
        random.seed(45)

        def make_objects():
            model = torch.nn.Linear(3, 2).double()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=.1, total_iters=6)
            return model, optimizer, scheduler

        def step(model, optimizer, scheduler):
            x = torch.randn(7, 3, dtype=torch.float64)
            y = torch.tensor(np.random.normal(size=(7, 2))) + random.random()
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.detach()

        model, optimizer, scheduler = make_objects()
        for _ in range(2):
            step(model, optimizer, scheduler)
        payload = {"model_state_dict": model.state_dict(), "epoch": 2}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            save_checkpoint_atomic(training_state(
                payload, optimizer, scheduler, {"epochs": 6}, [{"epoch": 2}], 1, .5), path)
            expected_loss = step(model, optimizer, scheduler)
            expected_parameters = copy.deepcopy(model.state_dict())
            expected_lr = optimizer.param_groups[0]["lr"]

            checkpoint = torch.load(path, weights_only=True)
            restored, restored_optimizer, restored_scheduler = make_objects()
            restored.load_state_dict(checkpoint["model_state_dict"])
            restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            restored_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
            state = checkpoint["numpy_rng_state"]
            np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                                 state["position"], state["has_gauss"], state["cached_gaussian"]))
            actual_loss = step(restored, restored_optimizer, restored_scheduler)
            torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
            for name, actual in restored.state_dict().items():
                torch.testing.assert_close(actual, expected_parameters[name], atol=0, rtol=0)
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], expected_lr)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
