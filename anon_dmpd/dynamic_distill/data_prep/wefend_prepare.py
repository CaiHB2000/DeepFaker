import argparse
from pathlib import Path

import pandas as pd


def load_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing WeFEND CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "Title": "title",
            "Report Content": "text_extra",
            "Image Url": "image_url",
            "label": "label_raw",
        }
    )
    text = df["title"].fillna("") + " " + df["text_extra"].fillna("")
    df["text"] = text.str.strip()
    df["label"] = df["label_raw"].astype(int)
    df["guid"] = df["News Url"].fillna(df["title"]).astype(str)
    df["image_url"] = df["image_url"].fillna("").astype(str)
    mask = df["text"].str.strip().astype(bool) & df["image_url"].str.strip().astype(bool)
    return df.loc[mask, ["guid", "text", "image_url", "label"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WeFEND train/test splits.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/wechat"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/wechat/processed"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": args.base_dir / "train" / "news.csv", "test": args.base_dir / "test" / "news.csv"}
    for split, csv_path in splits.items():
        df = load_split(csv_path)
        out_path = args.output_dir / f"wefend_{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(df)} rows).")


if __name__ == "__main__":
    main()
