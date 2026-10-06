"""Build the ProsQA splits used in the paper from the public COCONUT release.

The paper uses the ProsQA data of COCONUT (Hao et al., 2024), restricted to the
questions whose answer is at most 4 hops from the root (the public files also
contain 5- and 6-hop questions). This keeps 14,785 / 257 / 419 of the
17,886 / 300 / 500 train / valid / test questions.

    git clone https://github.com/facebookresearch/coconut   # data/prosqa_{train,valid,test}.json
    python build_prosqa.py --src coconut/data --out data
"""
import argparse
import json
import os

from data import shortest_path

MAX_HOPS = 4
KEEP = ("edges", "root", "target", "neg_target")   # the only fields the code reads; the text and the written-out reasoning steps of the release are dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory containing prosqa_{train,valid,test}.json")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "valid", "test"):
        with open(os.path.join(a.src, f"prosqa_{split}.json")) as f:
            raw = json.load(f)
        kept = [{k: r[k] for k in KEEP}
                for r in raw if len(shortest_path(r["edges"], r["root"], r["target"])) - 1 <= MAX_HOPS]
        with open(os.path.join(a.out, f"prosqa_{split}.json"), "w") as f:
            json.dump(kept, f)
        print(f"{split}: kept {len(kept)} of {len(raw)} questions -> {a.out}/prosqa_{split}.json")


if __name__ == "__main__":
    main()
