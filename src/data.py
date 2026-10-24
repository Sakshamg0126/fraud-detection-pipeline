import pandas as pd


def load_data(path, time_col="Time"):
    df = pd.read_csv(path)
    return df.sort_values(time_col).reset_index(drop=True)


def time_split(df, train_frac=0.6, val_frac=0.2):
    """Split chronologically into train / validation / test."""
    n = len(df)
    i = int(n * train_frac)
    j = int(n * (train_frac + val_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]