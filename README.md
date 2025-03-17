# Credit Score Classification: Random Forest & XGBoost

## 📌 Overview
This repository contains a **credit score classification project** using **Random Forest and XGBoost**. The goal is to predict a customer’s credit score category based on various financial and behavioral attributes. The project includes **data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and visualization**.

## 🚀 Models Used
1. **Random Forest** (Tuned, 77% Accuracy)
2. **XGBoost** (Tuned, 80% Accuracy)

## 📂 Project Structure
```
📦 Credit-Score-Classification
│-- 📁 data/               # Raw & processed datasets (not included in GitHub)
│-- 📁 notebooks/          # Jupyter Notebooks for exploration & model building
│-- 📁 models/             # Trained model files (if applicable)
│-- 📁 results/            # Model performance reports & visualizations
│-- 📜 Model_Comparison_Results.json # Detailed documentation
│-- 📜 README.md           # Project overview & instructions
│-- 📜 requirements.txt    # Required libraries
│-- 📜 train_model.py      # Python script for training the models
```

## 📊 Data Description
The dataset includes customer financial information such as **income, credit card usage, number of bank accounts, delayed payments, and credit inquiries**. The target variable is the **credit score category**:
- **0** → Poor
- **1** → Standard
- **2** → Good

## 🛠 Installation & Setup
To run this project locally, follow these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Credit-Score-Classification.git
cd Credit-Score-Classification
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Model Training
```bash
python train_model.py
```

### 4️⃣ Evaluate the Models
```bash
python evaluate_model.py
```

## 📈 Results & Model Performance
- **Random Forest Accuracy:** `77%`
- **XGBoost Accuracy:** `80%`
- **Cross-validation applied to prevent overfitting**
- **Feature importance analysis included**

| Model         | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|------------|--------|-----------|
| Random Forest | 77%     | 0.76      | 0.77   | 0.77      |
| XGBoost      | 80%     | 0.80      | 0.80   | 0.80      |

## 📌 Visual Summary
Here are some key visualizations from the project:

1️⃣ **Feature Correlation Heatmap**  
2️⃣ **Confusion Matrices for Both Models**  
3️⃣ **Model Performance Comparison**  

📊 ![Model Comparison](results/model_comparison.png)

## 💡 Key Takeaways
- **XGBoost slightly outperforms Random Forest** in this dataset.
- **Hyperparameter tuning significantly improved accuracy.**
- **Overfitting was managed using cross-validation & regularization techniques.**
- **SMOTE was used to handle class imbalance.**

## 🏆 Future Improvements
- Testing **deep learning models** (ANNs, CNNs for tabular data).
- Exploring **other feature engineering techniques**.
- Deploying the best model as an **API or web app**.

---

## 📝 Author
Deepak Shrivas  
📧 Email: dannyshrivas31@gmail.com 
🔗 LinkedIn: Deepak Shrivas(https://www.linkedin.com/in/deepak-shrivas)
🐙 GitHub: Danny0068(https://github.com/Danny0068)

---

## 🌟 Contribute & Support
If you found this project helpful, please ⭐ the repository and feel free to contribute by submitting a **Pull Request**! 🚀
