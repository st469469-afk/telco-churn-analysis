import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

root = Path(__file__).parent.parent.parent
model = joblib.load(root / "models" / "logreg.pkl")

X_test = pd.read_csv(root / "data" / "processed" / "X_test_processed.csv")

y_test = pd.read_csv(
    root / "data" / "processed" / "y_test.csv"
)["Churn"]

y_test = y_test.map({
    "No": 0,
    "Yes": 1
})

y_pred = model.predict(X_test)




report = classification_report(y_test, y_pred)

with open(root / "reports/classification_report.txt", "w") as f:
    f.write(report)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.savefig(root / "reports/confusion_matrix.png")
