
# 🛡️ Credit Card Fraud Detection using ML and Flask

This project is a full-stack machine learning application that detects fraudulent credit card transactions. It features a trained Random Forest model, preprocessing pipelines, a user-friendly Flask web interface, and visual performance reports.

---

© 2025 Vishal Benake. All rights reserved.  
Unauthorized copying, distribution, or modification of this code is prohibited.

---

## 🚀 Demo

Here are some screenshots of the live application:

### 🖼️ UI Demo

<p align="center">
  <img src="img/frontend-ui_page-1.jpg" alt="Frontend UI 1" width="400"/>
  <img src="img/frontend-ui_page-2.jpg" alt="Frontend UI 2" width="400"/>
</p>

### 🖼️ UI Demo Result Page

<p align="center">
  <img src="img/result-ui_page.jpg" alt="Result UI" width="800" height="500"/>
</p>

---

## 🧠 Machine Learning Overview

- **Model Used**: Random Forest Classifier  
- **Preprocessing**: Done using Scikit-learn Pipelines and SMOTE for imbalance  
- **Feature Handling**:
  - Scaling with `StandardScaler`
  - Handling class imbalance using SMOTE
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🗂️ Project Structure

```
Credit_Card_Fraud_Detection_Project/
│
├── app.py                        # Flask application
├── pipeline.py                   # Model training and pipeline
├── model.pkl                     # Trained pipeline model
├── requirements.txt              # Python dependencies
│
├── data/
│   └── creditcard.csv            # Dataset 
│
├── notebook/
│   ├── Credit_Card_Fraud_Detection_Project.ipynb
│   └── Credit_Card_Fraud_Detection_Project.pdf            
│
├── templates/
│   ├── index.html                # UI for input
│   └── result.html               # Result display
│
├── modules/
│   ├── model_evaluation.py
│   └── model_training.py
│
├── demo/
│   ├── frontend-ui_page-1.jpg
│   ├── frontend-ui_page-2.jpg 
│   └── result-ui_page.jpg
```
---

## 🛠️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/vishal-benake/Credit-Card-Fraud-Detection-using-ML-and-Flask.git
   cd Credit-Card-Fraud-Detection-ML-Flask
   ```

2. **Create virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**  
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000`

---

## 📈 Prediction Output

- The app predicts whether a credit card transaction is fraudulent or legitimate.
- A detailed result page shows:
  - Input feature breakdown
  - Confidence score (fraud probability)
  - User-friendly messages based on prediction

---

## 📌 Requirements

- Python ≥ 3.7  
- Flask  
- scikit-learn  
- pandas  
- imbalanced-learn  
- joblib  
- chart.js (for frontend charts)

Install via:
```bash
pip install -r requirements.txt
```

---

## <img src="https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_%282017%29.svg" width="20" height="20"> Youtube  
<h4>If you like, do follow me on Youtube</h4>  
<a href="https://www.youtube.com/@Code-With-Vishal">Connect with me on Youtube</a>

## <img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg" width="20" height="20"> Instagram  
<h4>If you like, do follow me on Instagram</h4>  
<a href="https://www.instagram.com/vishaal_87">Connect with me on Instagram</a>

---

© 2025 Vishal Benake. All rights reserved.  
Unauthorized copying, distribution, or modification of this code is prohibited.
