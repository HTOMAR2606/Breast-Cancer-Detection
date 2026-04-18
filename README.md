# ML-Based Breast Cancer Risk Prediction System

A modern **Machine Learning-powered Clinical Decision Support System** designed to classify breast cancer patients into **Low, Medium, and High Risk** categories based on tumor characteristics.

---

## 🚀 Live Demo

🔗 *[Add your deployed Streamlit link here]*

---

## 📌 Project Overview

This project uses machine learning techniques to analyze tumor morphological features and assist in **prioritizing high-risk breast cancer patients**.

It is designed as a **decision support tool**, not a diagnostic system, and aims to enhance clinical workflow by providing quick and consistent risk assessment.

---

## 🎯 Key Features

* 🧠 **Machine Learning Models**

  * Logistic Regression
  * Decision Tree
  * Random Forest (final model)

* 📊 **Risk Stratification**

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk

* 🖥️ **Premium UI Dashboard**

  * Dark-themed medical interface
  * Patient form + tumor input sliders
  * Real-time prediction

* 📄 **PDF Report Generation**

  * Patient details
  * Risk score & category
  * Visual risk representation

* ⚡ **Fast Processing**

  * Instant predictions (<200ms)

---

## 🧪 Input Features

The model uses key tumor characteristics:

* Tumor Size
* Surface Texture
* Tumor Boundary
* Tumor Area
* Smoothness Index

---

## 🧠 Machine Learning Workflow

1. Data Preprocessing (Scaling, Cleaning)
2. Model Training (Multiple Models)
3. Model Evaluation (Accuracy, ROC, Confusion Matrix)
4. Model Selection (Random Forest)
5. Deployment via Streamlit UI

---

## 📊 Tech Stack

* **Frontend/UI:** Streamlit + Custom CSS
* **Backend:** Python
* **ML Libraries:** Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **PDF Generation:** ReportLab

---

## 📁 Project Structure

```
├── app.py
├── data.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 🧠 Model Output

The system predicts:

* **Risk Score** (0–1 probability)
* **Risk Category:**

  * Low Risk
  * Medium Risk
  * High Risk

---

## ⚠️ Disclaimer

> This application is developed for **educational purposes only** and should not be used for real-world medical diagnosis. Clinical validation is required before deployment in healthcare environments.

---

## 🔮 Future Improvements

* Integration with Electronic Health Records (EHR)
* Use of larger clinical datasets (e.g., SEER)
* Deep learning for imaging data
* Model explainability (SHAP, LIME)

---

## 👨‍💻 Author

**Harsh**
B.Tech Student – Machine Learning Project

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

