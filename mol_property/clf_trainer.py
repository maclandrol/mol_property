# -*- coding: utf-8
import os
import numpy as np
import fsspec
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import pandas as pd
import click
from molfeat.trans.concat import FeatConcat
from molfeat.trans.fp import FPVecTransformer


def model_evaluation(model, x_input, y_input):
    y_pred = model.predict(x_input)
    print(y_pred.shape)
    print(classification_report(y_true=y_input, y_pred=y_pred))
    print(
        "f1_score [micro]: ", f1_score(y_true=y_input, y_pred=y_pred, average="micro")
    )
    print(
        "f1_score [macro]: ", f1_score(y_true=y_input, y_pred=y_pred, average="macro")
    )
    print(
        "precision_score [micro]: ",
        precision_score(y_true=y_input, y_pred=y_pred, average="micro"),
    )
    print(
        "precision_score [macro]: ",
        precision_score(y_true=y_input, y_pred=y_pred, average="macro"),
    )
    print(
        "recall_score [micro]: ",
        recall_score(y_true=y_input, y_pred=y_pred, average="micro"),
    )
    print(
        "recall_score [macro]: ",
        recall_score(y_true=y_input, y_pred=y_pred, average="macro"),
    )


@click.command()
@click.option("--data", required=True, help="Path to dataset")
@click.option("--output", required=True, help="Path to model output saving")
@click.option("--seed", type=int, default=7)
@click.option("--cv", type=int, default=5)
@click.option("--feat", multiple=True, type=str, default=["desc2D", "maccs", "estate"])
def main(data, output, seed, cv, feat):

    data = pd.read_csv(data)
    smiles = data["smiles"]
    all_feats = []
    for f in feat:
        ffn = FPVecTransformer(f)
        all_feats.append(ffn)
    cat_fp = FeatConcat(all_feats, dtype=float)
    X, _ = cat_fp(data["smiles"].values, ignore_errors=True)

    X = np.nan_to_num(X)
    train_idx = data[data.split != "test"].index.tolist()
    test_idx = data[data.split == "test"].index.tolist()
    y_data = data["Y"].values
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_test = y_data[test_idx]
    y_train = y_data[train_idx]

    print("\n ========================= \n")
    print("X_train.shape:", X_train.shape, "X_test.shape", X_test.shape)
    print("\n ========================= \n")

    xgb_model = xgb.XGBClassifier()

    params = {
        "estimator__max_depth": [4, 8, 10, 15],
        "estimator__objective": ["binary:logistic"],
        "estimator__n_estimators": [50, 100, 300],
        "estimator__subsample": [0.5, 0.75, 1.0],
        "estimator__colsample_bytree": [0.5, 0.75, 1.0],
        "estimator__eta": [1e-3, 0.01, 0.1, 0.2, 0.5],
        "estimator__booster": ["gbtree"],
    }

    clf = GridSearchCV(xgb_model, params, verbose=3, n_jobs=-1, cv=cv)
    clf.fit(X_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)

    print("train:", model_evaluation(clf.best_estimator_, X_train, y_train))
    print("test:", model_evaluation(clf.best_estimator_, X_test, y_test))

    clf.best_estimator_.save_model(output)


if __name__ == "__main__":
    main()