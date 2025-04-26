# Mobile Price Range Prediction

## Problem Statement
This project aims to predict the price range of mobile devices using machine learning techniques. The target variable is a categorical classification of price ranges (0-3), making this a multi-class classification problem.

## Dataset
**Source**: [train.csv](data/price_range/train.csv)  
**Features** (21 attributes):
- Technical specs: Battery Power, RAM, Screen Dimensions, Pixel Resolution, etc.
- Connectivity features: Bluetooth, WiFi, 3G/4G support
- Physical attributes: Weight, Dimensions
- Camera specs: Front/Back Camera resolution
- **Target**: Price_Range (0: Low cost, 3: High-end)

## Approach
1. **Data Preprocessing**:
   - No missing values or duplicates found
   - Feature engineering and transformation (StandardScaler, MinMaxScaler)
   - Correlation analysis and feature selection

2. **Models Implemented**:
   - Ordinal Logistic Regression
   - Ordinal Logistic Regression with Interaction Terms
   - Multinomial Logistic Regression
   - XGBoost (with Interaction Constraints)
   - Random Forest Classifier
   - Decision Tree Classifier

3. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - SHAP values for feature importance analysis

## Key Results
| Model                                  | Accuracy | F1-Score |
|----------------------------------------|----------|----------|
| Ordinal Logistic Regression            | 95.00%   | 95.01%   |
| Ordinal with Interaction Terms         | 94.75%   | 94.74%   |
| Multinomial Logistic Regression        | 92.75%   | 92.73%   |
| XGBoost (with Interaction Constraints) | 91.25%   | 91.26%   |
| Random Forest                          | 88.50%   | 88.41%   |
| Decision Tree                          | 87.00%   | 86.89%   |

**Top 5 Predictive Features** (SHAP analysis):
1. RAM
2. Battery Power
3. Pixel Resolution
4. Screen Size
5. Clock Speed

**Requirements**

    - Python 3.7+

    - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, shap

**How to Run?**

    - Clone the repository

    - Install dependencies:

        pip install -r requirements.txt

    - Update the dataset path inside the notebook

    - Run all cells in the Jupyter notebook sequentially

**Conclusion**

    - The Ordinal Logistic Regression model achieved the highest accuracy of 95%, showcasing strong predictive performance on mobile price ranges.

    - RAM was found to be the most influential feature for predicting the mobile price range, followed by Battery Power and Pixel Resolution.

    - Feature importance was validated using SHAP values.

**Future Enhancements**

    - Explore more complex models like Gradient Boosting (XGBoost, LightGBM) or Neural Networks.

    - Expand dataset size for better generalization.

    - Add richer features like brand information, market trends, or user ratings.

**Limitations**

    - Dataset limited to 2000 samples.

    - Model might not generalize well without more diverse and larger datasets.

## Code Structure
```bash
solution.ipynb
├── Data Collection & Exploration
├── Data Preprocessing
├── Feature Engineering
├── Model Training
├── Model Evaluation
└── SHAP Analysis
└── Conclusion


