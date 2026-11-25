from pathlib import Path

import pandas as pd


FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DATASET_PATH = Path(__file__).resolve().parent / "data" / "ai4i2020.csv"


def main():
    data = pd.read_csv(DATASET_PATH)
    active = data[FAILURE_MODES].eq(1)
    print(f"Total rows: {len(data)}")
    for label in FAILURE_MODES:
        count = int(active[label].sum())
        print(f"{label}: {count} positives, prevalence {count / len(data):.2%}")
    overlapping = int(active.sum(axis=1).ge(2).sum())
    print(f"Rows with two or more active failure-mode labels: {overlapping}")


if __name__ == "__main__":
    main()
