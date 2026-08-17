from .data_load import load_telco_churn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

df = load_telco_churn()

target_col = 'Churn'

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(inplace=True)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
cat_cols = [col for col in df.select_dtypes(include=["object"]).columns if col != 'Churn']

df.drop(columns=['customerID'], inplace=True)

df.drop_duplicates(inplace=True)

df.drop(columns=['gender', 'PhoneService'], inplace=True)

df['TotalCharges'] = np.log1p(df['TotalCharges'])

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X_train.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


from pathlib import Path
import pandas as pd
import numpy as np
import pickle



root_raw =  Path(__file__).parent.parent.parent / "data" / "raw"
root_processed =  Path(__file__).parent.parent.parent / "data" / "processed"



X.to_csv(root_raw / "X.csv", index=False)
y.to_frame(name="Churn").to_csv(root_raw / "y.csv", index=False)

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=preprocessor.get_feature_names_out()
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=preprocessor.get_feature_names_out()
)

X_train_processed.to_csv(root_processed / "X_train_processed.csv", index=False)
X_test_processed.to_csv(root_processed / "X_test_processed.csv", index=False)

y_train.to_csv(root_processed / "y_train.csv", index=False)
y_test.to_csv(root_processed/ "y_test.csv", index=False)

