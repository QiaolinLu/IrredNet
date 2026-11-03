import copy
import json
from pathlib import Path
import tempfile
import unittest

from irrednet.splits import SPLIT_VERSION, file_sha256, load_material_split


class TestExplicitMaterialSplits(unittest.TestCase):
    def setUp(self):
        self.samples = [{"JARVIS_ID": f"synthetic-{i}"} for i in range(5)]
        self.manifest = {
            "version": SPLIT_VERSION, "dataset_ids": [s["JARVIS_ID"] for s in self.samples],
            "splits": {"train": ["synthetic-3", "synthetic-1"], "val": ["synthetic-4"], "test": ["synthetic-0"]},
            "unused_ids": ["synthetic-2"], "seed": 32,
        }

    def load(self, manifest=None, samples=None, dataset_path=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            path.write_text(json.dumps(self.manifest if manifest is None else manifest))
            return load_material_split(path, self.samples if samples is None else samples,
                                       dataset_path=dataset_path)

    def test_ordered_ids_survive_dataset_reordering_and_keep_unused_sample(self):
        samples = list(reversed(self.samples))
        splits, provenance = self.load(samples=samples)
        for name, indices in splits.items():
            self.assertEqual([samples[i]["JARVIS_ID"] for i in indices], self.manifest["splits"][name])
        self.assertEqual(provenance["unused_ids"], ["synthetic-2"])
        self.assertEqual(provenance["num_unused"], 1)
        self.assertEqual(sum(map(len, splits.values())), 4)

    def test_overlap_duplicates_and_missing_materials_are_rejected(self):
        overlap = copy.deepcopy(self.manifest)
        overlap["splits"]["test"] = ["synthetic-1"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.load(overlap)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["splits"]["train"].append("synthetic-3")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(samples=self.samples + [self.samples[0]])
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.load(samples=self.samples[:-1])
        missing = copy.deepcopy(self.manifest)
        missing["unused_ids"] = []
        with self.assertRaisesRegex(ValueError, "cover the exact dataset"):
            self.load(missing)
        with self.assertRaisesRegex(ValueError, "needs a material ID"):
            self.load(samples=[{}] + self.samples[1:])

    def test_dataset_hash_guards_label_or_structure_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.pkl"
            path.write_bytes(b"original dataset content")
            manifest = copy.deepcopy(self.manifest)
            manifest["dataset_sha256"] = file_sha256(path)
            self.load(manifest, dataset_path=path)
            path.write_bytes(b"changed labels or structures")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.load(manifest, dataset_path=path)


if __name__ == "__main__":
    unittest.main()
