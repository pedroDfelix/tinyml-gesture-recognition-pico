"""Check that the C++ port gives the same results as the Python model (ai_puro.py).

Runs the original pure-Python forward pass on the same windows (20 readings,
step 5) and compares it with cpp_predictions.csv written by ./evaluate.
It also reports accuracy on the held-out 20% test split used for the confusion
matrix in the report (train_test_split, random_state=42, stratified).

Usage: python compare_with_python.py <repo folder> cpp_predictions.csv
Needs numpy, pandas and scikit-learn.
"""
import ast
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Circulo", "Quadrado", "Repouso", "Triangulo"]
FILES = ["dados_circulo.csv", "dados_quadrado.csv", "dados_repouso.csv", "dados_triangulo.csv"]
WINDOW, STEP = 20, 5


def load_weights(path):
    src = open(path, encoding="utf-8").read()
    return {n: np.array(ast.literal_eval(re.search(rf"^{n} = (.*)$", src, re.M).group(1)))
            for n in ["W1", "B1", "W2", "B2", "W3", "B3"]}


def forward(P, x):
    h1 = np.maximum(0, x @ P["W1"] + P["B1"])
    h2 = np.maximum(0, h1 @ P["W2"] + P["B2"])
    z = h2 @ P["W3"] + P["B3"]
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main(repo, cpp_csv):
    P = load_weights(f"{repo}/ai_puro.py")
    X, y = [], []
    for c, f in enumerate(FILES):
        a = pd.read_csv(f"{repo}/{f}")[["x", "y", "z"]].values.astype(float)
        for s in range(0, len(a) - WINDOW + 1, STEP):
            w = a[s:s + WINDOW]
            X.append(((w - w[0]) / 500.0).reshape(-1))
            y.append(c)
    X, y = np.array(X), np.array(y)
    probs_py = forward(P, X)
    pred_py = probs_py.argmax(1)

    cpp = pd.read_csv(cpp_csv)
    if len(cpp) != len(y) or not (cpp["class"].values == y).all():
        sys.exit("window lists do not match")
    pred_cpp = cpp["pred_cpp"].values
    probs_cpp = cpp[["p0", "p1", "p2", "p3"]].values

    same = (pred_cpp == pred_py).sum()
    print(f"Same predicted class: {same}/{len(y)} windows")
    print(f"Largest probability difference: {np.abs(probs_cpp - probs_py).max():.2e}")

    idx = np.arange(len(y))
    _, test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    for name, pred in [("Python", pred_py), ("C++", pred_cpp)]:
        acc = (pred[test] == y[test]).mean()
        print(f"{name} accuracy on the 20% test split: {acc:.2%} ({(pred[test] == y[test]).sum()}/{len(test)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
