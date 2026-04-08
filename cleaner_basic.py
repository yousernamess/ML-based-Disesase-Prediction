import pandas as pd

INPUT_FILE = "sympdis.csv"
OUTPUT_FILE = "sympdis_clean_basic.csv"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    # First column is disease label (kept as-is).
    disease_col = df.columns[0]
    symptom_cols = df.columns[1:]

    print("Initial shape:", df.shape)
    print("Disease column kept as-is:", disease_col)

    # 1) Remove exact duplicate rows.
    duplicate_rows = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print("\nDuplicates removed:", duplicate_rows)
    print("Shape after duplicate removal:", df.shape)

    # 2) Validate symptom values are only 0/1.
    invalid_mask = ~df[symptom_cols].isin([0, 1])
    invalid_rows = invalid_mask.any(axis=1).sum()
    print("\nRows with invalid symptom values (not 0/1):", invalid_rows)

    if invalid_rows > 0:
        invalid_counts = invalid_mask.sum()
        invalid_counts = invalid_counts[invalid_counts > 0].sort_values(ascending=False)
        print("Columns containing invalid values:")
        print(invalid_counts)

        # Basic rule: drop rows with any invalid symptom values.
        df = df.loc[~invalid_mask.any(axis=1)].reset_index(drop=True)
        print("Shape after dropping invalid rows:", df.shape)
    else:
        print("All symptom columns passed 0/1 validation.")

    # 3) Drop constant symptom columns (all 0 or all 1).
    constant_symptom_cols = [
        col for col in symptom_cols if df[col].nunique(dropna=False) <= 1
    ]
    df = df.drop(columns=constant_symptom_cols)

    print("\nConstant symptom columns dropped:", len(constant_symptom_cols))
    if constant_symptom_cols:
        print("First constant columns:", constant_symptom_cols[:20])
        if len(constant_symptom_cols) > 20:
            print(f"... and {len(constant_symptom_cols) - 20} more")

    print("Final shape:", df.shape)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
