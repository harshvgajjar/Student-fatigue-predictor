# 🎓 Predicting Digital Fatigue in Remote Learners Using Machine Learning

> **MSc Data Science Dissertation** — University of East London (2025–2026)  
> A multi-model machine learning approach to detect digital fatigue from Virtual Learning Environment interaction logs.

---

## 📌 Overview

Digital fatigue — cognitive exhaustion from prolonged engagement with digital learning platforms — is a growing threat to student wellbeing and academic performance. Despite its prevalence, **no prior study had applied machine learning to predict digital fatigue directly from VLE interaction logs**.

This project addresses that gap by building and comparing four machine learning models trained on real-world educational data, achieving an **AUC of 0.9763** with XGBoost.

---

## 📊 Datasets

| Dataset | Size | Description |
|---|---|---|
| **OULAD** (Open University Learning Analytics) | 1.8M sessions, 32,593 students | VLE interaction logs, demographics, assessment results |
| **EdNet KT4** | 500 users (sample) | Knowledge tracing sequential interaction data |

---

## 🔧 Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras, SHAP, Matplotlib, Seaborn
- **Models:** Logistic Regression, Random Forest, XGBoost, LSTM
- **Framework:** CRISP-DM
- **Techniques:** Feature Engineering, SHAP Interpretability, Fairness Evaluation, Dimensionality Reduction

---

## 🚀 What I Built

### 1. Proxy Fatigue Label Construction
Since no direct fatigue label existed in the data, I engineered a **binary fatigue label** from six behavioural signals extracted from VLE interaction logs (e.g. click patterns, session duration, task-switching ratio).

### 2. Feature Engineering
Constructed a **39-feature set** from raw interaction logs covering:
- Session-level behaviour (clicks, unique sites visited, task-switch ratio)
- Temporal patterns (time of day, week of module)
- Content engagement types (OU content, forums, quizzes)

### 3. Model Training & Comparison
Trained and compared four models:

| Model | AUC | F1 Score | Precision |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.9487 | — | — |
| Random Forest | 0.9635 | — | — |
| **XGBoost (Best)** | **0.9763** | **0.8723** | **0.8979** |
| LSTM (EdNet KT4) | 0.6812 | — | — |

> ✅ XGBoost reduced false fatigue alerts by **63%** compared to Logistic Regression.

### 4. SHAP Interpretability
Applied **SHAP (SHapley Additive Explanations)** to explain model predictions — identifying OU content engagement as the dominant fatigue predictor.

### 5. Fairness Evaluation
Conducted an **ethical fairness audit** across gender, disability status, and deprivation index to ensure equitable model performance across student demographics.

---

## 📈 Key Results

- 🏆 **XGBoost achieved AUC = 0.9763** — strongest overall performance
- 📉 63% reduction in false positive fatigue alerts vs baseline
- ⚖️ Fair performance confirmed across gender, disability, and deprivation dimensions
- 🔍 SHAP identified activity-type features as dominant fatigue predictors

---

## 📁 Project Structure

```
student-fatigue-predictor/
│
├── data/                  # Dataset references (OULAD & EdNet KT4)
├── notebooks/             # Jupyter notebooks for each pipeline stage
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_fairness_evaluation.ipynb
├── models/                # Saved model artefacts
├── reports/               # Figures and evaluation outputs
└── README.md
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/harshvgajjar/student-fatigue-predictor.git
cd student-fatigue-predictor

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
```

**Dependencies:** pandas, numpy, scikit-learn, xgboost, tensorflow, shap, matplotlib, seaborn

---

## 🔮 Future Work

- Transformer-based sequential models (e.g. BERT for education)
- Real-time fatigue detection dashboard
- Federated learning for privacy-preserving prediction
- Direct fatigue measurement via physiological signals

---

## 👨‍💻 Author

**Harsh Gajjar**  
MSc Data Science — University of East London  
📧 harshvgajjar@gmail.com  
🔗 https://www.linkedin.com/in/harsh-gajjar-3b317926b/

---
## 🌐 Live Demo
https://student-fatigue-predictor.streamlit.app/

## 📄 Supervisor

**Dr Rita Obatolu** — University of East London

---

*This project was completed as part of the DS7010 Dissertation module, Academic Year 2025–2026.*
