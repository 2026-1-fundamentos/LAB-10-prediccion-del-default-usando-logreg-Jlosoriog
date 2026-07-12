import gzip
import json
import os
import pickle

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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def clean_dataset(df):
    df = df.copy()

    df = df.rename(
        columns={"default payment next month": "default"}
    )

    df = df.drop(columns=["ID"])

    df = df.dropna()

    # Tratamiento estándar utilizado en este ejercicio
    df["EDUCATION"] = df["EDUCATION"].replace(
        [0, 5, 6],
        4,
    )

    df["MARRIAGE"] = df["MARRIAGE"].replace(
        [0],
        3,
    )

    return df


train_df = pd.read_csv(
    "files/input/train_data/train_default_of_credit_card_clients.csv"
)

test_df = pd.read_csv(
    "files/input/test_data/test_default_of_credit_card_clients.csv"
)

train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)

x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]

x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]

categorical_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

numerical_features = [
    col
    for col in x_train.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "num",
            MinMaxScaler(),
            numerical_features,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "select_k_best",
            SelectKBest(
                score_func=f_classif
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

param_grid = {
    "select_k_best__k": [
        10,
        12,
        14,
        16,
        18,
        20,
        "all",
    ],
    "classifier__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100,
    ],
    "classifier__solver": [
        "liblinear",
    ],
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

model.fit(x_train, y_train)

print("Best score:", model.best_score_)
print("Best params:", model.best_params_)
print("Train score:", model.score(x_train, y_train))
print("Test score:", model.score(x_test, y_test))

os.makedirs(
    "files/models",
    exist_ok=True,
)

os.makedirs(
    "files/output",
    exist_ok=True,
)

with gzip.open(
    "files/models/model.pkl.gz",
    "wb",
) as file:
    pickle.dump(
        model,
        file,
    )


def compute_metrics(
    y_true,
    y_pred,
    dataset_name,
):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def compute_confusion_matrix(
    y_true,
    y_pred,
    dataset_name,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

metrics_train = compute_metrics(
    y_train,
    y_train_pred,
    "train",
)

metrics_test = compute_metrics(
    y_test,
    y_test_pred,
    "test",
)

cm_train = compute_confusion_matrix(
    y_train,
    y_train_pred,
    "train",
)

cm_test = compute_confusion_matrix(
    y_test,
    y_test_pred,
    "test",
)

with open(
    "files/output/metrics.json",
    "w",
    encoding="utf-8",
) as file:
    for record in [
        metrics_train,
        metrics_test,
        cm_train,
        cm_test,
    ]:
        file.write(
            json.dumps(record) + "\n"
        )