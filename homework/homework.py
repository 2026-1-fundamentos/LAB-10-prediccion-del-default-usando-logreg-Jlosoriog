import gzip
import json
import pickle
import os
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

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    # Elimina registros con EDUCATION o MARRIAGE == 0 (N/A)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    # Agrupa EDUCATION > 4 en la categoria "others" (4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df


train_df = pd.read_csv("files/input/train_data/train_default_of_credit_card_clients.csv")
test_df = pd.read_csv("files/input/test_data/test_default_of_credit_card_clients.csv")

train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)

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
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", MinMaxScaler(), numerical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("select_k_best", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

param_grid = {
    "select_k_best__k": [10, 15, 20, "all"],
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"],
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

os.makedirs("files/models", exist_ok=True)
os.makedirs("files/output", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

def compute_metrics(y_true, y_pred, dataset_name):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_cm(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
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


metrics_train = compute_metrics(y_train, y_train_pred, "train")
metrics_test = compute_metrics(y_test, y_test_pred, "test")
cm_train = compute_cm(y_train, y_train_pred, "train")
cm_test = compute_cm(y_test, y_test_pred, "test")

with open("files/output/metrics.json", "w") as f:
    for record in [metrics_train, metrics_test, cm_train, cm_test]:
        f.write(json.dumps(record) + "\n")

print("Listo. Mejor balanced_accuracy (CV):", model.best_score_)
print("Mejores parametros:", model.best_params_)
print(metrics_train)
print(metrics_test)