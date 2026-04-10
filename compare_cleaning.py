import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_DEFAULT = "sympdis.csv"
CLEANED_CANDIDATES = ["sympdis_clean.csv"]


def pick_existing(path_candidates: Iterable[str]) -> str:
    for candidate in path_candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any cleaned dataset file. Looked for: "
        + ", ".join(path_candidates)
    )


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} must contain a disease column and at least one symptom column")
    return df


def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def invalid_symptom_rows(df: pd.DataFrame) -> tuple[int, pd.Series]:
    symptom_cols = df.columns[1:]
    invalid_mask = ~df[symptom_cols].isin([0, 1])
    rows = invalid_mask.any(axis=1)
    invalid_counts = invalid_mask.sum()
    invalid_counts = invalid_counts[invalid_counts > 0].sort_values(ascending=False)
    return int(rows.sum()), invalid_counts


def constant_symptom_columns(df: pd.DataFrame) -> list[str]:
    symptom_cols = df.columns[1:]
    return [col for col in symptom_cols if df[col].nunique(dropna=False) <= 1]


def missing_values_total(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_top_counts(series: pd.Series, limit: int = 10) -> None:
    if series.empty:
        print("None")
        return
    for name, value in series.head(limit).items():
        print(f"- {name}: {int(value)}")


def disease_counts(df: pd.DataFrame) -> pd.Series:
    return df.iloc[:, 0].value_counts()


def top_symptom_prevalence(df: pd.DataFrame, limit: int = 10) -> pd.Series:
    symptom_cols = df.columns[1:]
    prevalence = df[symptom_cols].mean().sort_values(ascending=False)
    return prevalence.head(limit)


def report_dataset(name: str, df: pd.DataFrame) -> None:
    print_header(f"{name} DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Disease column: {df.columns[0]}")
    print(f"Symptom columns: {df.shape[1] - 1}")
    print(f"Missing values total: {missing_values_total(df)}")
    print(f"Duplicate rows: {count_duplicate_rows(df)}")

    invalid_rows, invalid_counts = invalid_symptom_rows(df)
    print(f"Rows with invalid symptom values: {invalid_rows}")
    if invalid_rows:
        print("Columns with invalid values:")
        print_top_counts(invalid_counts)

    constant_cols = constant_symptom_columns(df)
    print(f"Constant symptom columns: {len(constant_cols)}")
    if constant_cols:
        print("First constant symptom columns:")
        for col in constant_cols[:20]:
            print(f"- {col}")

    labels = disease_counts(df)
    print(f"Unique diseases: {labels.shape[0]}")
    print("Top 10 diseases by sample count:")
    print_top_counts(labels, 10)

    symptoms = top_symptom_prevalence(df, 10)
    print("Top 10 symptom prevalences:")
    for name, value in symptoms.items():
        print(f"- {name}: {value:.6f} ({value * 100:.2f}%)")


def compare_datasets(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    print_header("CLEANING COMPARISON")

    raw_rows, raw_cols = raw.shape
    clean_rows, clean_cols = cleaned.shape
    print(f"Rows removed overall: {raw_rows - clean_rows}")
    print(f"Columns removed overall: {raw_cols - clean_cols}")

    raw_dup = count_duplicate_rows(raw)
    clean_dup = count_duplicate_rows(cleaned)
    print("\n1) Duplicate row removal")
    print(f"- Raw duplicate rows: {raw_dup}")
    print(f"- Cleaned duplicate rows: {clean_dup}")
    print(f"- Duplicate rows removed: {raw_dup - clean_dup}")

    raw_invalid_rows, raw_invalid_counts = invalid_symptom_rows(raw)
    clean_invalid_rows, clean_invalid_counts = invalid_symptom_rows(cleaned)
    print("\n2) Invalid value filtering")
    print(f"- Raw rows with invalid symptom values: {raw_invalid_rows}")
    print(f"- Cleaned rows with invalid symptom values: {clean_invalid_rows}")
    print(f"- Invalid rows removed: {raw_invalid_rows - clean_invalid_rows}")
    if raw_invalid_rows:
        print("- Raw columns most affected by invalid values:")
        print_top_counts(raw_invalid_counts)

    raw_constant = constant_symptom_columns(raw)
    clean_constant = constant_symptom_columns(cleaned)
    raw_constant_set = set(raw_constant)
    clean_constant_set = set(clean_constant)
    removed_constant_cols = sorted(raw_constant_set - clean_constant_set)
    print("\n3) Constant symptom column removal")
    print(f"- Raw constant symptom columns: {len(raw_constant)}")
    print(f"- Cleaned constant symptom columns: {len(clean_constant)}")
    print(f"- Constant columns removed: {len(removed_constant_cols)}")
    if removed_constant_cols:
        print("- First removed constant columns:")
        for col in removed_constant_cols[:20]:
            print(f"  - {col}")

    raw_diseases = disease_counts(raw)
    clean_diseases = disease_counts(cleaned)
    print("\n4) Disease label stability")
    print(f"- Unique diseases in raw: {raw_diseases.shape[0]}")
    print(f"- Unique diseases in cleaned: {clean_diseases.shape[0]}")
    missing_in_cleaned = sorted(set(raw_diseases.index) - set(clean_diseases.index))
    print(f"- Disease labels missing after cleaning: {len(missing_in_cleaned)}")
    if missing_in_cleaned:
        for name in missing_in_cleaned[:20]:
            print(f"  - {name}")

    print("\n5) Symptom prevalence shift")
    raw_symptoms = raw.columns[1:]
    clean_symptoms = cleaned.columns[1:]
    removed_symptoms = sorted(set(raw_symptoms) - set(clean_symptoms))
    print(f"- Symptom columns in raw: {len(raw_symptoms)}")
    print(f"- Symptom columns in cleaned: {len(clean_symptoms)}")
    print(f"- Symptom columns removed: {len(removed_symptoms)}")
    if removed_symptoms:
        print("- First removed symptom columns:")
        for col in removed_symptoms[:20]:
            print(f"  - {col}")

    print("\n6) Data retention")
    print(f"- Raw rows kept: {clean_rows / raw_rows:.4f} of original" if raw_rows else "- Raw rows kept: n/a")
    print(f"- Raw columns kept: {clean_cols / raw_cols:.4f} of original" if raw_cols else "- Raw columns kept: n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw and cleaned symptom-to-disease datasets.")
    parser.add_argument("--raw", default=RAW_DEFAULT, help="Path to the raw dataset CSV")
    parser.add_argument(
        "--cleaned",
        default="",
        help="Path to the cleaned dataset CSV. If omitted, the script looks for common cleaned filenames.",
    )
    args = parser.parse_args()

    raw_path = args.raw
    cleaned_path = args.cleaned or pick_existing(CLEANED_CANDIDATES)

    raw_df = load_dataset(raw_path)
    cleaned_df = load_dataset(cleaned_path)

    print(f"Raw dataset: {raw_path}")
    print(f"Cleaned dataset: {cleaned_path}")

    report_dataset("RAW", raw_df)
    report_dataset("CLEANED", cleaned_df)
    compare_datasets(raw_df, cleaned_df)


if __name__ == "__main__":
    main()
