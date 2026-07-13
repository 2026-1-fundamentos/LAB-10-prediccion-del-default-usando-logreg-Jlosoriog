import gzip
import json
import os
import pickle as pkl
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def clean_data(df):
    df = df.copy()
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "EDUCATION" in df.columns and "MARRIAGE" in df.columns:
        df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    df = df.dropna()
    return df


def run_pipeline():
    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", MinMaxScaler(), numerical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )

    params = {
        "feature_selection__k": [1],
        "classifier__C": [1],
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pkl.dump(model, file)

    os.makedirs("files/output", exist_ok=True)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        {
            "type": "metrics",
            "dataset": "train",
            "precision": float(
                precision_score(y_train, y_train_pred, zero_division=0)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_train, y_train_pred)
            ),
            "recall": float(
                recall_score(y_train, y_train_pred, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_train, y_train_pred, zero_division=0)
            ),
        },
        {
            "type": "metrics",
            "dataset": "test",
            "precision": float(
                precision_score(y_test, y_test_pred, zero_division=0)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, y_test_pred)
            ),
            "recall": float(
                recall_score(y_test, y_test_pred, zero_division=0)
            ),
            "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
        },
    ]

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metrics.append(
        {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {
                "predicted_0": int(cm_train[0, 0]),
                "predicted_1": int(cm_train[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm_train[1, 0]),
                "predicted_1": int(cm_train[1, 1]),
            },
        }
    )

    metrics.append(
        {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {
                "predicted_0": int(cm_test[0, 0]),
                "predicted_1": int(cm_test[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm_test[1, 0]),
                "predicted_1": int(cm_test[1, 1]),
            },
        }
    )

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for row in metrics:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    run_pipeline()