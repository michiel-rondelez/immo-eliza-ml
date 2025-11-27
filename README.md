# Immo Eliza ML

A machine learning project to predict real estate prices in Belgium for Immo Eliza.

## Description

This project builds upon previously scraped and analyzed Belgian real estate data to create a predictive model for property prices. The pipeline includes data preprocessing, feature engineering, model training, and evaluation using multiple regression techniques.

## Installation

```bash
# Clone the repository
git clone https://github.com/michiel-rondelez/immo-eliza-ml.git
cd immo-eliza-ml

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
with poetry.lock

## Usage

### Run main

```bash
python main.py
```

## Project Structure

```
immo-eliza-ml/
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── models/
├── plots/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── main.py
│   └── predict.py

```

## Data Pipeline

### 1. Data Cleaning
- Handling duplicates
- Managing missing values
- Dropping irrelevant columns/rows

### 2. Preprocessing
- **NaN Handling**: Imputation strategies for missing values
- **Encoding**: One-hot encoding for categorical features
- **Scaling**: Standardization of numeric features
- **Feature Selection**: Correlation analysis to select relevant features

### 3. Model Training
- Train/test split
- Model fitting and serialization

### 4. Evaluation
- Performance metrics: R², MSE, MAE
- Overfitting analysis
## Results

### Key Findings

- *Feature importance insights*
This I didn't find out
- *Model comparison results*

  ------------------------------------------------------------
            model  train_r2  test_r2  test_rmse  test_mae
Linear Regression  0.712231 0.672048   0.316557  0.234856
              SVR  0.812560 0.697372   0.304089  0.225438
    Decision Tree  0.699821 0.584221   0.356433  0.268343
    Random Forest  0.741191 0.645944   0.328914  0.243410
          XGBoost  0.805598 0.727746   0.288425  0.213862
------------------------------------------------------------
- *Overfitting analysis*
- 
Model                  Train R²    Test R²        Gap          Status
-----------------------------------------------------------------
Linear Regression        0.7122     0.6720     0.0402            OK ✅
SVR                      0.8126     0.6974     0.1152      OVERFIT ⚠️
Decision Tree            0.6998     0.5842     0.1156      OVERFIT ⚠️
Random Forest            0.7412     0.6459     0.0952      MODERATE ⚡
XGBoost                  0.8056     0.7277     0.0779      MODERATE ⚡

## Timeline

- **Duration**: 4 days
- **Deadline**: 27/11/2024 5:00 PM
- **Presentation**: 01/12/2024 9:30 - 10:30 AM

## Contributors

- Michiel Rondelez

## License

This project is part of the BeCode AI training program.
