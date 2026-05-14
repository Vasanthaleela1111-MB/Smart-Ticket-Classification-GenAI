# 🤖 AI-Powered Customer Support Automation

## 📌 Project Overview

This project focuses on automating customer support ticket classification using Machine Learning, Deep Learning (LSTM), and Generative AI. The system classifies incoming support tickets into appropriate categories and generates automated responses using Gemini API.

---

## 🎯 Problem Statement

Organizations receive thousands of customer support tickets daily. Manual classification leads to delays, inefficiency, and poor customer experience.
This project aims to:

* Automatically classify support tickets
* Route them to the correct department
* Generate polite automated responses

---

## 🚀 Features

* 🔍 Text preprocessing (cleaning, tokenization, stopword removal)
* 📊 Multiple ML models:

  * Logistic Regression
  * Naive Bayes
  * Support Vector Machine (SVM)
  * Random Forest Classifier
* 🧠 Deep Learning model (LSTM)
* 🤖 Generative AI integration (Gemini API)
* 📈 Model evaluation (Accuracy, Precision, Recall, F1-score)
* ⚖️ Handling class imbalance
* 🌐 Streamlit web app (optional deployment)

---

## 🧠 Models Used

### 🔹 Machine Learning Models

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

### 🔹 Deep Learning

* LSTM (Many-to-One RNN for sequence classification)

---

## ⚙️ Tech Stack

* Python 🐍
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* Pandas, NumPy
* Streamlit (for deployment)
* Google Gemini API

---

## 📂 Dataset

* Source: Hugging Face
* Dataset: `Tobi-Bueck/customer-support-tickets`

### Fields:

* `body` → Ticket text
* `queue` → Target category
* Optional: priority, tags, subject

---

## 🔄 Project Workflow

```text
Data Collection
   ↓
Text Preprocessing
   ↓
TF-IDF (for ML models)
   ↓
Train-Test Split
   ↓
Model Training (ML + LSTM)
   ↓
Model Evaluation
   ↓
Gemini API Integration
   ↓
Streamlit Deployment
```

---

## 📊 Results

* ML Models Accuracy: Moderate
* LSTM Model Accuracy: Improved performance
* Handles multi-class classification (52 categories)
* Performance affected by class imbalance (handled using class weights)

---

## 🔥 Key Learnings

* NLP preprocessing techniques
* Feature extraction (TF-IDF)
* Sequence modeling using LSTM
* Handling class imbalance
* Integrating ML with Generative AI
* Building end-to-end AI applications

---

## 🤖 Gemini API Integration

After classification:

1. Predict ticket category
2. Send ticket + category to Gemini API
3. Generate a polite automated response

---

## 📦 Installation

```bash
git clone https://github.com/your-username/AI-Customer-Support-Automation.git
cd AI-Customer-Support-Automation
```

---

## ▶️ Usage

```bash
python geminiui.py
```

Or run Streamlit app:

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

* Improve accuracy using BERT/RoBERTa
* Add sentiment analysis
* Better UI/UX for application
* Real-time API deployment

---

## 👩‍💻 Author

**Vasantha Leela M**

* LinkedIn: https://www.linkedin.com/in/vasantha-leela-mb-3b7009362
* GitHub: https://github.com/Vasanthaleela1111-MB

---

## ⭐ Acknowledgment

Dataset provided by Hugging Face
Inspired by real-world customer support automation challenges

---
