import pandas as pd
import kagglehub
from pathlib import Path

import shutil



def load_telco_churn() -> pd.DataFrame:
    """
    Завантажує датасет Telco Customer Churn з Kaggle.
    Копіює CSV у data/raw і повертає DataFrame.
    """
    root = Path(__file__).parent.parent  
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
    source_csv = cache_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    target_csv = raw_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not target_csv.exists():
        shutil.copy2(source_csv, target_csv)

    df = pd.read_csv(target_csv)
    return df
