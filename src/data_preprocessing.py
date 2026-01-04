import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
