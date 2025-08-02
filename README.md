# 💳 Credit Card Fraud Detection using Logistic Regression

## 🧠 Project Overview

This project focuses on building a machine learning model using **Logistic Regression** to detect **fraudulent credit card transactions**. The dataset used is highly imbalanced, making it a perfect use case to explore class balancing techniques and proper evaluation metrics beyond accuracy.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Instances**: 284,807 transactions
- **Classes**: 
  - `0`: Legitimate
  - `1`: Fraudulent
- **Features**:
  - 28 anonymized principal components (`V1` to `V28`)
  - `Time`, `Amount`
  - `Class` (Target variable)

> ⚠️ The dataset is extremely imbalanced: ~0.17% of transactions are fraudulent.

---

## 🛠️ Technologies Used

- Python 3.11+
- NumPy
- Pandas
- scikit-learn
- imbalanced-learn (for SMOTE)
- Matplotlib & Seaborn (for visualization)

---

## 🔁 Project Workflow

### 1. Data Preprocessing
- Load and inspect the dataset
- Apply `StandardScaler` for scaling `Amount` and `Time` features
- Address class imbalance using `SMOTE`

### 2. Model Training
- Train a Logistic Regression model
- Tune hyperparameters like `max_iter`, `solver`, and regularization strength

### 3. Model Evaluation
- Visualize performance with accuracy
---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-logistic.git
cd fraud-detection-logistic

# Install dependencies
pip install -r requirements.txt

# Run the main script
python fraud_detection.py
