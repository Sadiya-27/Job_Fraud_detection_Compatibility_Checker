import streamlit as st
import pandas as pd
import re
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import base64
import os

# -------------------------
# Preprocessing
# -------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -------------------------
# Load Data and Train Model
# -------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("fake_job_postings.csv")
    data = data[["title", "description", "fraudulent"]].dropna()
    data["text"] = (data["title"] + " " + data["description"]).apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data["text"])
    y = data["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🕵️ Job Post Fraud Detection + 🎯 Compatibility Checker")

st.markdown("Upload your resume or enter skills manually to check compatibility with a job post and predict fraud risk.")

st.markdown(f"📈 **Fraud Detection Model Accuracy:** `{accuracy * 100:.2f}%`")

title = st.text_input("Job Title")
description = st.text_area("Job Description")
job_skills_input = st.text_area("Required Job Skills (comma-separated)", placeholder="e.g. python, sql, tableau")

uploaded_resume = st.file_uploader("📄 Upload Your Resume (.pdf or .txt)", type=["pdf", "txt"])
manual_skills = st.text_area("Or enter your skills manually (comma-separated)", placeholder="e.g. python, data analysis, ML")

def safe_text(text):
    return re.sub(r"[^\x00-\xFF]", "", str(text))  # Remove anything outside Latin-1



def generate_report(title, description, fraud_prob, sim, top_resume, top_job, matched_skills, missing_skills, match_percent):
    def clean(text):
        return ''.join(c if ord(c) < 256 else '' for c in str(text))

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(70, 130, 180)  # Steel Blue
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 12, "Job Post Fraud Detection & Compatibility Report", ln=True, align='C', fill=True)
    pdf.ln(8)
    pdf.cell(0, 10, f"Model Accuracy: {accuracy * 100:.2f}%", ln=True)


    # Reset text settings
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)

    # Job Information
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(230, 230, 250)  # Lavender
    pdf.cell(0, 10, "Job Information", ln=True, fill=True)

    pdf.set_font("Arial", '', 12)
    pdf.cell(40, 10, "Title:", ln=False)
    pdf.cell(0, 10, clean(title), ln=True)
    pdf.multi_cell(0, 10, f"Description:\n{clean(description)}")
    pdf.ln(5)

    # Summary Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(245, 245, 245)  # Light Gray
    pdf.cell(0, 10, "Summary", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Fraud Probability: {fraud_prob * 100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Compatibility Score: {sim * 100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Skill Match: {match_percent:.2f}%", ln=True)
    pdf.ln(5)

    # Divider
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Matched / Missing Skills
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Matched Skills:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, clean(", ".join(matched_skills)) if matched_skills else "None")

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Missing Required Skills:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, clean(", ".join(missing_skills)) if missing_skills else "None")
    pdf.ln(5)

    # Top Resume Skills
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top Skills from Resume:", ln=True)
    pdf.set_font("Arial", '', 12)
    for skill, score in top_resume:
        pdf.cell(0, 8, f"{clean(skill)}: {score:.2f}", ln=True)

    # Top Job Skills
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top Required Job Skills:", ln=True)
    pdf.set_font("Arial", '', 12)
    for skill, score in top_job:
        pdf.cell(0, 8, f"{clean(skill)}: {score:.2f}", ln=True)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 10, "Generated using Job Compatibility & Fraud Detection Tool", align='C')

    report_path = "report.pdf"
    pdf.output(report_path)
    return report_path


if st.button("Check for Fraud & Compatibility"):
    if not title or not description:
        st.warning("Please enter both title and description.")
    else:
        job_text = clean_text(title + " " + description)
        job_vector = vectorizer.transform([job_text])

        pred = model.predict(job_vector)[0]
        fraud_prob = model.predict_proba(job_vector)[0][1]

        if pred == 1:
            st.error(f"⚠️ Likely Fraudulent (Confidence: {fraud_prob * 100:.2f}%)")
        else:
            st.success(f"✅ Likely Legitimate (Confidence: {(1 - fraud_prob) * 100:.2f}%)")

        resume_text = ""
        if uploaded_resume:
            if uploaded_resume.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_resume)
            elif uploaded_resume.type == "text/plain":
                resume_text = uploaded_resume.read().decode("utf-8")
            resume_text = clean_text(resume_text)
            st.success("✅ Resume uploaded and parsed.")
        elif manual_skills:
            resume_text = clean_text(manual_skills)
        else:
            st.warning("Upload resume or enter skills.")
            st.stop()

        # Cosine similarity between job description and resume
                # Extract cleaned skill lists
        job_skills = [s.strip().lower() for s in job_skills_input.split(",") if s.strip()]
        if manual_skills:
            resume_skills = [s.strip().lower() for s in manual_skills.split(",") if s.strip()]
        else:
            resume_skills = list(set(re.findall(r'\b[a-zA-Z]+\b', resume_text.lower())))

        # Combine both lists to create a corpus
        job_doc = " ".join(job_skills)
        resume_doc = " ".join(resume_skills)
        corpus = [job_doc, resume_doc]

        # TF-IDF Vectorization
        skill_vectorizer = TfidfVectorizer()
        vectors = skill_vectorizer.fit_transform(corpus)
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        st.info(f"🎯 Improved Compatibility Score: **{sim * 100:.2f}%**")

        # TF-IDF skills
        # TF-IDF skills - fixed
        feature_names = skill_vectorizer.get_feature_names_out()
        job_tfidf = vectors[0].toarray()[0]
        resume_tfidf = vectors[1].toarray()[0]

        top_job = sorted(zip(feature_names, job_tfidf), key=lambda x: x[1], reverse=True)[:10]
        top_resume = sorted(zip(feature_names, resume_tfidf), key=lambda x: x[1], reverse=True)[:10]

        # Bar Chart
        st.markdown("### 🔍 Visual Comparison")
        labels = [s[0] for s in top_job[:5]]
        job_vals = [s[1] for s in top_job[:5]]
        resume_vals = [dict(top_resume).get(label, 0) for label in labels]

        fig, ax = plt.subplots()
        bar_width = 0.35
        index = range(len(labels))
        ax.bar(index, job_vals, bar_width, label='Job')
        ax.bar([i + bar_width for i in index], resume_vals, bar_width, label='Resume')
        ax.set_xticks([i + bar_width/2 for i in index])
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("TF-IDF Score")
        ax.set_title("Top Skill Match")
        ax.legend()
        st.pyplot(fig)

        # Skill Matching
        job_skills = [s.strip().lower() for s in job_skills_input.split(",") if s.strip()]

        if manual_skills:
            resume_skills = [s.strip().lower() for s in manual_skills.split(",") if s.strip()]
        else:
    # fallback: extract words from resume text assuming it's raw text
            resume_skills = list(set(re.findall(r'\b[a-zA-Z]+\b', resume_text.lower())))


        matched_skills = list(set(job_skills) & set(resume_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))
        match_percent = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0

        st.markdown("### 🧠 Skill Match Summary")
        st.info(f"✅ Skill Match: **{match_percent:.2f}%**")
        st.markdown("**✅ Matched Skills:**")
        st.write(", ".join(matched_skills) if matched_skills else "_None_")
        st.markdown("**❌ Missing Skills:**")
        st.write(", ".join(missing_skills) if missing_skills else "_None_")

        # PDF Report
        report_path = generate_report(title, description, fraud_prob, sim, top_resume, top_job, matched_skills, missing_skills, match_percent)
        with open(report_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="Job_Report.pdf">📥 Download Detailed Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        os.remove(report_path)

st.markdown("---")
st.caption("Resume parsing powered by PyPDF2. Model trained on Kaggle's Fake Job Postings dataset.")
