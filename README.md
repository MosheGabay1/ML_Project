# 🏡 Housing Price Forecasting Pipeline

> A complete machine learning workflow to estimate house sale prices using structured data and gradient-boosted models.

---

## 📋 Project Overview

This repository contains an end-to-end solution for predicting residential property sale prices based on a variety of features. The main stages consist of:

1. **🔍 Initial Data Exploration (EDA)**  
2. **🧹 Cleaning & Imputation of Missing Values**  
3. **⚙️ Feature Creation & Transformation**  
4. **🌲 Model Development & Tuning** (XGBoost & LightGBM)  
5. **✅ Model Evaluation & Selection**  
6. **📈 Generating Final Price Predictions**

**Chosen Final Model:**  LightGBM (selected for its lowest cross-validated RMSE and efficient handling of categorical inputs)

---

## ✨ Key Highlights

- **🛠️ Comprehensive Feature Engineering**  
  - Developed 20+ derived attributes (e.g., combined bathroom count, total living area, property age indicator)  
  - Applied log transformations to reduce skewness in price and area variables  

- **🔄 Full-Scale Pipeline**  
  - Steps: Data ingestion → Cleaning → Feature construction → Model training → Prediction  
  - Ensured reproducibility with fixed random seeds and documented preprocessing steps  

- **📊 Model Comparison**  
  - Tested **XGBoost** (with/without PCA) against **LightGBM**  
  - LightGBM achieved ~10% lower RMSLE (on log-transformed target) compared to XGBoost  

- **🔍 Robust Missing-Value Strategy**  
  - Combined median/mode imputation for simple columns  
  - Employed KNN-based imputation for numeric features with moderate missing rates  

---

## ℹ️ Data Description

- **Data File:** `train_sample.csv`  
- **Core Columns:**  
  - **Location & Zoning:** `Neighborhood`, `MSZoning`  
  - **Lot & Land Details:** `LotArea`, `LotFrontage`  
  - **Building Specs:** `YearBuilt`, `YearRemodAdd`, `TotalBsmtSF`, `GrLivArea`  
  - **Rooms & Amenities:** `FullBath`, `HalfBath`, `BedroomAbvGr`, `GarageCars`  
  - **Quality & Condition:** `OverallQual`, `OverallCond`  
  - **Target Variable:** `SalePrice`  

- **Missing Data Patterns:**  
  - Numeric features (e.g., `LotFrontage`) filled with neighborhood medians  
  - Categorical features (e.g., `GarageType`) replaced with `"None"` or `"Missing"`  
  - Dropped columns where >75% of entries were null  

---

## 🛠️ Technical Details

### 1. Data Cleaning & Preprocessing
- **Invalid Entries:** Replaced unrealistic values (negative areas, implausible year built) with `NaN`.  
- **Imputation Approach:**  
  - **Numeric Columns:** Used median imputation for features like `LotFrontage`.  
  - **Categorical Columns:** Filled blanks with a placeholder category.  
  - **KNN Imputation:** Applied for moderate missing numeric attributes to leverage neighbor information.  
- **Outlier Treatment:** Identified extreme points in `GrLivArea` vs. `SalePrice` and removed top 1% as outliers.  

### 2. Feature Engineering
- **New Variables:**  
  - `TotalBaths` = `FullBath` + 0.5 × `HalfBath` + `BsmtFullBath` + 0.5 × `BsmtHalfBath`  
  - `TotalLivingArea` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`  
  - `PropertyAge` = `YrSold` − `YearBuilt`  
  - `IsRemodeled` = 1 if `YearRemodAdd` > `YearBuilt`, else 0  
  - Boolean flags: `HasPool`, `HasFireplace`, `HasGarage`  
- **Transformations:**  
  - **Log(1 + x)** for skewed numeric features (e.g., `SalePrice`, `GrLivArea`)  
  - **One-Hot Encoding** for nominal categories with low-to-medium cardinality (`Neighborhood`, `Exterior1st`)  
  - **Label Encoding** for ordered categories (`ExterQual`, `BsmtExposure`)  

### 3. Model Development & Tuning
- **Algorithms Trained:**  
  1. **XGBoost (All Features, No PCA)**  
     - Hyperparameters:  
       - `n_estimators=800`, `learning_rate=0.05`, `max_depth=6`  
       - `subsample=0.8`, `colsample_bytree=0.8`  
     - Employed early stopping on a validation fold to avoid overfitting.  
  2. **XGBoost (With PCA)**  
     - Standardized numeric inputs, applied PCA (95% variance retained)  
     - Trained XGBoost on principal components to test dimensionality reduction benefits.  
  3. **LightGBM (Final Choice)**  
     - Hyperparameters:  
       - `num_leaves=31`, `learning_rate=0.05`, `n_estimators=800`  
       - `boosting_type='gbdt'`, `objective='regression'`  
     - Handled categorical splits natively, reducing memory usage and speeding up training.  

- **Validation Strategy:**  
  - **K-Fold Cross-Validation (5 folds)** on training data  
  - Evaluated **RMSE** on log-transformed `SalePrice` for consistent comparison across models  

---

## 📈 Results & Insights

### Model Performance (5-Fold CV RMSE on Log SalePrice)
| Model                        | Avg. RMSE  |
|------------------------------|------------|
| XGBoost (No PCA)             | 0.343      |
| XGBoost (With PCA)           | 0.303      |
| **LightGBM (Final Model)**   | **0.206**  |

> LightGBM yielded the lowest RMSE, demonstrating superior accuracy in predicting (log) sale prices.

### Top Features (LightGBM Importance)
1. **OverallQual** – Overall quality of the house materials and finish  
2. **GrLivArea** – Above-ground living area (in sqft)  
3. **TotalBaths** – Combined count of full and half bathrooms  
4. **PropertyAge** – Age of the property at sale  
5. **GarageCars** – Number of garage spaces  
6. Additional strong predictors: `TotalLivingArea`, `Fireplaces`, `YearBuilt`, `LotArea`, etc.

> These variables carried the most weight in the final LightGBM predictions.

---

## 🚀 Final Predictions

- **Retrained** LightGBM on the full training set using optimal hyperparameters.  
- **Applied** the final model to a held-out test dataset to produce predicted sale prices.  
- **Output:** A table or CSV file with columns (`Id`, `PredictedSalePrice`) for each record in the test data.

---

## 📚 How to Use This Pipeline

1. **Requirements:**  
   - Python 3.7+  
   - Libraries:  
     ```
     pandas
     numpy
     scikit-learn
     xgboost
     lightgbm
     matplotlib
     seaborn
     ```
2. **Run the Notebook:**  
   - Open `Final_Work_ML_Project.ipynb` in Jupyter (or VSCode with the Python extension).  
   - Execute the cells in sequence: Data loading → Cleaning → Feature Engineering → Model Training → Evaluation → Prediction.  
   - Modify hyperparameters or feature selections to experiment further.

---

## 📝 License

Distributed under the **MIT License**. Feel free to use, adapt, and distribute as you see fit.

---

## ✉️ Contact

For questions or suggestions, please reach out:

- **Author:** Moshe Gabay  
- **Email:** moshe.gabay@example.com  

Happy modeling! 🚀  
