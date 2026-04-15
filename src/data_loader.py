import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    # si colonne target existe
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        # fallback: dernière colonne = label
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )

    return X_train, X_test, y_train, y_test
