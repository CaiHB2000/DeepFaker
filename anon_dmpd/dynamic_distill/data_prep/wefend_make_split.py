import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(row: pd.Series) -> str:
    title = str(row.get("Title", ""))
    content = str(row.get("Report Content", ""))
    text = (title + " " + content).strip()
    return " ".join(text.split())


def load_wefend_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df.apply(normalize_text, axis=1)
    df = df[df["text"].astype(bool)].copy()
    df["label"] = df["label"].astype(int)
    df["guid"] = df["News Url"].fillna(df["Title"]).astype(str)
    df["account"] = df["Ofiicial Account Name"].astype(str)
    df["image_url"] = df["Image Url"].fillna("")
    return df


def has_image(image_dir: Path, guid: str) -> bool:
    digest = hashlib.sha256(guid.encode("utf-8")).hexdigest()
    hashed = image_dir / f"{digest}.jpg"
    if hashed.exists():
        return True
    return (image_dir / f"{guid}.jpg").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified WeFEND splits with text deduplication.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/wechat"))
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/wechat/images"))
    parser.add_argument("--train-path", type=Path, default=Path("train/news.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("test/news.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/wechat/processed_split"))
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base = args.base_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_wefend_csv(base / args.train_path)
    test_df = load_wefend_csv(base / args.test_path)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Remove duplicate texts and duplicate GUIDs to avoid leakage.
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["guid"]).reset_index(drop=True)

    # Keep only entries with downloaded images.
    combined = combined[combined["guid"].apply(lambda g: has_image(image_dir, g))].reset_index(drop=True)
    if combined.empty:
        raise RuntimeError("No WeFEND samples with available images.")

    # First hold out test portion.
    temp_df, test_split = train_test_split(
        combined,
        test_size=args.test_size,
        stratify=combined["label"],
        random_state=args.random_state,
    )

    # Split remaining into train/val.
    val_size = args.val_size / (1 - args.test_size)
    train_split, val_split = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=temp_df["label"],
        random_state=args.random_state,
    )

    def save(df: pd.DataFrame, name: str) -> None:
        out_path = output_dir / f"wefend_{name}.csv"
        df[["guid", "text", "image_url", "label"]].to_csv(out_path, index=False)
        print(
            f"Saved {out_path} ({len(df)} rows) label distribution:\n{df['label'].value_counts()}"
        )

    save(train_split, "train")
    save(val_split, "val")
    save(test_split, "test")


if __name__ == "__main__":
    main()
