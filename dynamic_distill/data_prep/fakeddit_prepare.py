import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"id", "clean_title", "title", "image_url", "2_way_label", "3_way_label", "6_way_label"}


def load_split(base_dir: Path, split_tag: str, label_scheme: int = 2) -> pd.DataFrame:
    tsv_path = base_dir / "multimodal_only_samples" / f"multimodal_{split_tag}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing TSV file: {tsv_path}")

    df = pd.read_table(
        tsv_path,
        engine="python",
        on_bad_lines="skip",
        dtype=str,
        keep_default_na=False,
    )

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Required columns {missing_cols} not found in {tsv_path}")

    df["text"] = df["clean_title"].replace("", pd.NA).fillna(df["title"]).fillna("")
    if label_scheme == 2:
        col = "2_way_label"
    elif label_scheme == 3:
        col = "3_way_label"
    elif label_scheme == 6:
        col = "6_way_label"
    else:
        raise ValueError(f"Unsupported label_scheme={label_scheme}; choose from 2/3/6.")
    df["label"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["guid"] = df["id"].astype(str)
    df["image_url"] = df["image_url"].fillna("").astype(str)

    mask = df["text"].str.strip().astype(bool) & df["image_url"].str.strip().astype(bool)
    df = df.loc[mask, ["guid", "text", "image_url", "label"]].reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Fakeddit multimodal CSV splits.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/fakeddit/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/fakeddit/processed"))
    parser.add_argument("--label-scheme", type=int, choices=[2, 3, 6], default=2,
                        help="Choose label scheme: 2 (binary), 3, or 6 classes.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"train": "train", "val": "validate", "test": "test_public"}
    # organize by subfolder per scheme for clarity
    scheme_dir = args.output_dir / {2: "2way", 3: "3way", 6: "6way"}[args.label_scheme]
    scheme_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_tag in split_map.items():
        df = load_split(args.base_dir, split_tag, label_scheme=args.label_scheme)
        out_csv = scheme_dir / f"fakeddit_{split_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} with {len(df)} samples (label_scheme={args.label_scheme}).")


if __name__ == "__main__":
    main()
