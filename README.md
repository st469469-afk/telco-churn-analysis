# Customer Churn Prediction

This repository contains a data science project focused on analyzing customer churn and building a predictive model to identify customers who are likely to leave a telecom service.

## Project objectives

- Analyze customer behavior and service usage to understand key drivers of churn  
- Build and evaluate machine learning models for churn prediction  
- Provide business insights and recommendations to reduce churn and improve retention

## Dataset

- **Source:** Telco customer churn dataset (e.g., Kaggle Telco Customer Churn)  
- **Target variable:** `Churn` (Yes / No)  
- **Main feature groups:**
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - Contract & billing: `Contract`, `PaperlessBilling`, `PaymentMethod`
  - Financials & tenure: `tenure`, `MonthlyCharges`, `TotalCharges`

The raw data is stored in the `data/` directory (not tracked in this repo if it is large or private).

## Repository structure

```text
telco-churn-analysis/
├─ data/           # Raw and processed data (usually ignored by git)
├─ notebooks/      # Jupyter notebooks with EDA and experiments
├─ src/            # Source code (preprocessing, training, evaluation)
├─ models/         # Saved trained models 
├─ reports/        # Generated reports and figures 
├─ main.py         # Main script / entry point 
└─ README.md
```

## Methodology

1. **Exploratory Data Analysis (EDA)**  
The exploratory data analysis includes a comprehensive dataset description, target variable analysis, and both univariate and bivariate analyses of customer features with respect to churn.

2. **Data preprocessing**  
   Handling missing values and data types , encoding categorical features (one‑hot encoding), train–test split, and feature scaling where needed.

3. **Modeling**  
   Logistic Regression, evaluation using accuracy, precision, recall, F1‑score

4. **Model interpretation**  
   Feature importance from logistic regression coefficients, analysis of which features increase or decrease churn risk.

## Key results

- The model performs better at identifying customers who **do not churn** (class 0), with high precision and moderate recall.  
- For **churned customers** (class 1), recall is relatively strong (the model captures most at‑risk customers), but precision is lower, leading to a noticeable number of false positives.  
- The main weakness is **low precision for the churn class**, which can trigger unnecessary retention efforts for customers who would not have churned.  
- The most influential features for churn prediction are:
  - **Contract type** (especially month‑to‑month)
  - **MonthlyCharges** and **TotalCharges**
  - **InternetService** type (fiber optic vs DSL)
  - **Tenure** (customer lifetime)
  - **PaymentMethod** (e.g., electronic check)
  - Availability of **OnlineSecurity** and **TechSupport**

These patterns indicate that new customers on short‑term, expensive plans with fewer support/security services are at the highest risk of churn.

## How to run

1. **Clone the repository**

```bash
git clone <your_repo_url>.git
cd telco-churn-analysis
```

2. **Create and activate virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Prepare data**

- Place the dataset file (e.g., `Telco-Customer-Churn.csv`) into the `data/` directory.  
- Update paths in notebooks or `src/` scripts if needed.

5. **Run notebooks or main script**

- EDA and modeling: open notebooks from `notebooks/` and run them in order.  
- Or run the pipeline (if implemented) via:

```bash
python main.py
```

## Business insights and recommendations

- Focus retention efforts on customers with **month‑to‑month contracts** and **high monthly charges**, and on customers with **fiber optic** internet and **short tenure**.  
- Encourage at‑risk customers to switch to **longer‑term contracts** (one‑year, two‑year) and add **OnlineSecurity**, **TechSupport**, and other value‑adding services.  
- Review the pricing structure for high‑charge plans to reduce churn incentives.

## Next steps

- Experiment with more advanced models (e.g., Random Forest, XGBoost) and compare performance.  
- Implement probability‑based thresholds to better balance recall and precision for churned customers.  
- Deploy the model as a simple API or dashboard to support real‑time retention campaigns.
