# Job_Fraud_detection_Compatibility_Checker

# 🕵️‍♀️ Job Fraud Detection & 🎯 Compatibility Checker

This Streamlit web application helps users:
- Detect whether a job post is fraudulent using a machine learning model.
- Evaluate how compatible their resume or skills are with the job description.
- Generate a downloadable PDF report containing fraud prediction, skill match analysis, and TF-IDF skill comparisons.

## 🔍 Features

- ✅ **Fraud Detection** using a Logistic Regression model trained on the Kaggle Fake Job Postings dataset.
- 📄 **Resume Parsing** via PDF/Text upload or manual skill entry.
- 🎯 **Compatibility Checker** using TF-IDF + Cosine Similarity to score job-post vs. resume match.
- 📊 **Top Skills Comparison** using dynamic charts.
- 📋 **PDF Report Generator** summarizing all results.
- 🔐 Runs completely locally and does **not store** any uploaded resumes or personal data.

---

## 🗂️ Directory Structure

📁 Job_Fraud_detection_Compatibility_Checker/
│
├── app.py # Main Streamlit app
├── fake_job_postings.csv # Dataset used for training
├── requirements.txt # Python dependencies
├── report.pdf # (Optional) Example generated report
└── README.md # This file

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sadiya-27/Job_Fraud_detection_Compatibility_Checker.git
cd Job_Fraud_detection_Compatibility_Checker
```

### 2. Install Dependencies
Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
### 3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```
Then open the browser at: http://localhost:8501

---

# 🧠 Model Details
- Algorithm: Logistic Regression

- Feature Extraction: TF-IDF (Top 1000 features)

- Target: Binary Classification (Fraudulent: 1, Legitimate: 0)

- Dataset: Kaggle Fake Job Postings Dataset: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

---

# 📦 Dependencies
Key libraries used:

1. streamlit – Web app framework

2. sklearn – Machine Learning and TF-IDF

3. PyPDF2 – PDF parsing

4. matplotlib – Plotting bar charts

5. fpdf – PDF report generation

6. pandas, re, base64, os – General utilities

Install them via:

```bash
pip install streamlit scikit-learn PyPDF2 matplotlib fpdf pandas
```

# 📁 Deployment on Streamlit Cloud
- Push this project to a GitHub repository.

- Go to https://streamlit.io/cloud

- Click "New app" > Connect your GitHub > Select this repo

- Set Main file path to app.py

- Make sure requirements.txt exists.

- Click "Deploy"

- Your app will be live shortly!

---

# ✍️ Example Use Case
- You’re a job seeker unsure about the legitimacy of a remote job post.

- You paste the job description into the app, upload your resume.

- The app checks:

  - Whether the post seems fraudulent.

  - How well your resume matches it.

  - Missing vs. matched skills.

- Provides a professional PDF report.
  
---

## 📄 License
This project is open-source under the MIT License.

---

## 🙌 Acknowledgments
- Kaggle for the Fake Job Postings Dataset: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction 

- Streamlit for providing an easy platform to create data apps.

---

