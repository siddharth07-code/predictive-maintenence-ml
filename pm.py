import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier

data = pd.read_csv("data/ai4i2020.csv")

print(data.head())
print("Shape:", data.shape)

print(data.info())

print(data.isnull().sum())

print(data["Machine failure"].value_counts())

sns.countplot(x="Machine failure", data=data)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Machines")
plt.show()

data = data.drop(
    ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

data = pd.get_dummies(
    data,
    columns=["Type"],
    dtype=int
)

print(data.head())
print(data.columns)

X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

print("Features:")
print(X.head())

print("Target:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = logistic_model.predict(
    X_test_scaled
)

print("\nLogistic Regression")
print(classification_report(y_test, y_pred_lr))

decision_tree = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

decision_tree.fit(
    X_train,
    y_train
)

y_pred_dt = decision_tree.predict(
    X_test
)

print("\nDecision Tree")
print(classification_report(y_test, y_pred_dt))

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(
    X_train,
    y_train
)

y_pred_rf = random_forest.predict(
    X_test
)

print("\nRandom Forest")
print(classification_report(y_test, y_pred_rf))

X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

X_train_xgb.columns = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "type_H",
    "type_L",
    "type_M"
]

X_test_xgb.columns = X_train_xgb.columns

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_xgb,
    y_train
)

y_pred_xgb = xgb_model.predict(
    X_test_xgb
)

print("\nXGBoost")
print(classification_report(y_test, y_pred_xgb))

results = {
    "Logistic Regression": [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr)
    ],
    "Decision Tree": [
        accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_dt)
    ],
    "Random Forest": [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf)
    ],
    "XGBoost": [
        accuracy_score(y_test, y_pred_xgb),
        precision_score(y_test, y_pred_xgb),
        recall_score(y_test, y_pred_xgb),
        f1_score(y_test, y_pred_xgb)
    ]
}

results_df = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nModel Comparison")
print(results_df)

importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10, 5))
importance.plot(kind="bar")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

joblib.dump(logistic_model, "logistic_regression.pkl")
joblib.dump(decision_tree, "decision_tree.pkl")
joblib.dump(random_forest, "random_forest.pkl")
joblib.dump(xgb_model, "xgboost.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\nAll models saved successfully.")