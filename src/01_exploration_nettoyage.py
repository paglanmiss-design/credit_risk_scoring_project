from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "credit_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_data.csv"
GRAPH_PATH = ROOT / "graphs" / "debt_to_income_distribution.png"
GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw CSV file and infer the target column if the label uses a variant."""
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            cleaned = line.strip().strip('"').strip("'")
            rows.append([part.strip().strip('"').strip("'") for part in cleaned.split(",")])

    if not rows:
        raise ValueError(f"No rows were found in {path}")

    header = rows[0]
    data_rows = rows[1:]
    df = pd.DataFrame(data_rows, columns=header)

    for column in df.columns:
        if column.lower() in {"revenu", "income"}:
            df.rename(columns={column: "income"}, inplace=True)
        elif column.lower() in {"anciennete_emploi", "employment_years"}:
            df.rename(columns={column: "employment_years"}, inplace=True)
        elif column.lower() in {"endettement", "debt_to_income_ratio"}:
            df.rename(columns={column: "debt_to_income_ratio"}, inplace=True)
        elif column.lower() in {"nb_credits_anterieurs", "previous_loans"}:
            df.rename(columns={column: "previous_loans"}, inplace=True)
        elif column.lower() in {"defaut", "default", "target"}:
            df.rename(columns={column: "default"}, inplace=True)

    for column in df.columns:
        if column in {"income", "employment_years", "debt_to_income_ratio", "previous_loans", "age"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "default" in df.columns:
        df["default"] = pd.to_numeric(df["default"], errors="coerce")

    return df


def detect_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing duplicates, imputing missing values, and filtering obvious outliers."""
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    if "default" in numeric_columns:
        numeric_columns.remove("default")

    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    if numeric_columns:
        for column in numeric_columns:
            q1 = cleaned[column].quantile(0.25)
            q3 = cleaned[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            cleaned = cleaned[(cleaned[column] >= lower_bound) & (cleaned[column] <= upper_bound)]

    return cleaned


def summarize_dataset(df: pd.DataFrame) -> None:
    print("=== Credit risk dataset exploration ===")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes)

    print("\nDescriptive statistics for financial variables:")
    financial_vars = [col for col in df.columns if col != "default" and pd.api.types.is_numeric_dtype(df[col])]
    stats = df[financial_vars].describe().T[["mean", "50%", "min", "max", "std"]].rename(columns={"50%": "median"})
    print(stats)

    if "default" in df.columns:
        print("\nClass balance for the target variable 'default':")
        print(df["default"].value_counts(dropna=False))
        print("\nClass proportions:")
        print(df["default"].value_counts(normalize=True).round(4))
    else:
        print("Target column 'default' was not found.")

    print("\nMissing values per column:")
    print(df.isna().sum())


def plot_debt_to_income_distribution(df: pd.DataFrame) -> None:
    debt_column = "debt_to_income_ratio"
    if debt_column not in df.columns:
        debt_column = next((col for col in df.columns if "debt" in col.lower() or "endett" in col.lower()), None)

    if debt_column is None:
        raise ValueError("No debt-to-income column was found in the dataset.")

    plt.figure(figsize=(8, 4))
    plt.hist(df[debt_column].dropna(), bins=10, color="#2E86AB", edgecolor="black")
    plt.title("Debt-to-income ratio distribution")
    plt.xlabel("Debt-to-income ratio")
    plt.ylabel("Number of clients")
    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()
    print(f"\nDistribution plot saved to {GRAPH_PATH}")


if __name__ == "__main__":
    df = load_dataset(RAW_DATA_PATH)
    summarize_dataset(df)
    cleaned_df = detect_and_clean(df)
    plot_debt_to_income_distribution(cleaned_df)

    cleaned_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\nCleaned dataset saved to {CLEAN_DATA_PATH}")