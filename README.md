# Startup Success & Failure Prediction

A multiclass machine learning project that predicts startup outcomes as **Failure, Acquisition, or IPO** using business, financial, founder, and market-related features.

## Problem Statement

The goal is to identify patterns associated with startup outcomes and build a classification model that predicts:

- Failure
- Acquisition
- IPO

The dataset contains **100,000 startup records and 11 columns**.

## Dataset

Source: [Kaggle — Startup Funding and Outcome Dataset](https://www.kaggle.com/datasets/dhrubangtalukdar/startup-funding-and-outcome-dataset)

The dataset contains **100,000 startup records and 11 columns**.

### Numerical Features

- `funding_rounds`
- `founder_experience_years`
- `team_size`
- `market_size_billion`
- `product_traction_users`
- `burn_rate_million`
- `revenue_million`

### Categorical Features

- `investor_type`
- `sector`
- `founder_background`

### Target

- `outcome`

Target distribution:

| Outcome | Percentage |
|---|---:|
| Failure | 55.61% |
| Acquisition | 42.34% |
| IPO | 2.06% |

## Data Cleaning & Preparation

The dataset was audited before modeling.

- No missing values
- No duplicate records
- Data types were appropriate
- Categorical values were checked for consistency
- No unnecessary columns were identified
- Unusual numerical values were investigated rather than automatically removed
- Potential data leakage was considered, particularly for `revenue_million` and `product_traction_users`

No values were removed simply because they appeared unusual.

## Exploratory Data Analysis

The main EDA questions focused on:

1. Outcome distribution
2. Revenue differences across outcomes
3. Investor type vs outcome
4. Numerical feature relationships
5. Product traction vs revenue

### Key Findings

- IPO represents only **2.06%** of the dataset.
- IPO startups generally have higher revenue than Failure and Acquisition startups.
- Product traction and revenue have a strong positive correlation (**0.714**).
- High traction combined with high revenue is more strongly associated with IPO.
- Investor type shows very little difference in outcome distribution.
- Revenue and product traction were among the most important predictive features.

## Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
