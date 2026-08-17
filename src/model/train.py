from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import pickle
import pandas as pd

root = Path(__file__).parent.parent.parent

X_train = pd.read_csv(root / "data" / "processed" / "X_train_processed.csv")

y_train = pd.read_csv(
    root / "data" / "processed" / "y_train.csv"
)["Churn"]

y_train = y_train.map({
    "No": 0,
    "Yes": 1
})

log_reg_cv_model = LogisticRegressionCV(
    solver="liblinear",
    cv=10,
    scoring="f1",
    n_jobs=-1,
    class_weight="balanced"
)

log_reg_cv_model.fit(X_train, y_train)

with open(root / "models" / "logreg.pkl", "wb") as f:
    pickle.dump(log_reg_cv_model, f)


