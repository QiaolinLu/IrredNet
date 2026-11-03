"""Print explicit O(3) -> space-group branching for demos or existing samples."""

import argparse
import json

import numpy as np

from .data import load_samples, sample_structure
from .decomposition import PROPERTY_FORMULAS, decompose_property


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Simple cubic Pm-3m crystal")
    source.add_argument("--dataset", help="Read-only preprocessed project pickle")
    source.add_argument("--structure", help="Structure file supported by pymatgen, e.g. CIF/POSCAR")
    parser.add_argument("--target", choices=["all", *PROPERTY_FORMULAS], default="all")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--symprec", type=float, default=1e-5)
    args = parser.parse_args()
    if args.demo:
        structure = (np.eye(3) * 4, [[0, 0, 0]], [14])
    elif args.structure:
        from pymatgen.core import Structure
        structure = Structure.from_file(args.structure)
    else:
        samples = load_samples(args.dataset)
        structure = sample_structure(samples[args.index])
    targets = PROPERTY_FORMULAS if args.target == "all" else [args.target]
    print(json.dumps([decompose_property(structure, target, symprec=args.symprec).summary()
                      for target in targets], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
