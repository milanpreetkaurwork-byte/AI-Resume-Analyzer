import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# AI RESUME ANALYZER
# ==============================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer & Job Recommendation System")
st.subheader("IBM SkillsBuild AI Internship Project")

st.write(
    "Upload your resume to extract skills, match suitable job roles "
    "and receive AI-based recommendations."
)

# ==============================
# JOB DATASET
# ==============================

jobs = {
    "Business Analyst": [
        "python", "sql", "excel", "statistics",
        "business analysis", "communication", "powerpoint"
    ],
    "Data Analyst": [
        "python", "sql", "excel", "pandas",
        "numpy", "statistics", "data analysis"
    ],
    "Data Scientist": [
        "python", "pandas", "numpy",
        "statistics", "machine learning"
    ],
    "Operations Analyst": [
        "excel", "statistics", "communication",
        "business analysis"
    ],
    "Finance & Business Analyst": [
        "excel", "finance", "accounting",
        "financial analysis", "financial modeling"
    ],
    "HR Analyst": [
        "excel", "communication",
        "statistics", "business analysis"
    ],
    "AI/ML Intern": [
        "python", "machine learning",
        "numpy", "pandas", "statistics"
    ],
    "Project Coordinator": [
        "communication", "excel",
        "powerpoint", "project management"
    ],
    "Marketing Analyst": [
        "excel", "statistics",
        "communication", "data analysis"
    ],
    "Financial Analyst": [
        "excel", "finance", "accounting",
        "financial analysis", "financial modeling"
    ]
}

# ==============================
# SKILL EXTRACTION
# ==============================

all_skills = sorted(
    set(skill for skills in jobs.values() for skill in skills)
)

def extract_skills(text):
    text = text.lower()
    detected = []

    for skill in all_skills:
        if skill.lower() in text:
            detected.append(skill)

    return detected


# ==============================
# RESUME TEXT EXTRACTION
# ==============================

def read_resume(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode(
            "utf-8", errors="ignore"
        )

    elif file_name.endswith(".docx"):
        from docx import Document

        document = Document(uploaded_file)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
        ]

        return "\n".join(paragraphs)

    else:
        return ""


# ==============================
# RESUME UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "📄 Upload your Resume",
    type=["txt", "docx"]
)

if uploaded_file:

    resume_text = read_resume(uploaded_file)

    st.success(
        f"Resume uploaded successfully: {uploaded_file.name}"
    )

    # ==============================
    # SKILLS
    # ==============================

    detected_skills = extract_skills(resume_text)

    st.header("🎯 Skills Detected")

    if detected_skills:
        st.write(
            ", ".join(
                skill.title()
                for skill in detected_skills
            )
        )

        st.metric(
            "Total Skills Detected",
            len(detected_skills)
        )

    else:
        st.warning("No matching skills detected.")

    # ==============================
    # JOB MATCHING
    # ==============================

    results = []

    for job_role, required_skills in jobs.items():

        matched = [
            skill
            for skill in required_skills
            if skill in detected_skills
        ]

        match_score = (
            len(matched) / len(required_skills) * 100
        )

        job_text = " ".join(required_skills)

        vectorizer = TfidfVectorizer()

        tfidf = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        similarity = cosine_similarity(
            tfidf[0:1],
            tfidf[1:2]
        )[0][0] * 100

        final_score = (
            0.60 * match_score +
            0.40 * similarity
        )

        results.append({
            "Job Role": job_role,
            "Match Score": round(match_score, 2),
            "NLP Similarity": round(similarity, 2),
            "Final AI Score": round(final_score, 2)
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Final AI Score",
        ascending=False
    ).reset_index(drop=True)

    # ==============================
    # BEST RECOMMENDATION
    # ==============================

    best_job = results_df.iloc[0]

    st.header("🏆 Best Job Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Job Role",
            best_job["Job Role"]
        )

    with col2:
        st.metric(
            "Final AI Score",
            f'{best_job["Final AI Score"]}%'
        )

    with col3:
        st.metric(
            "NLP Similarity",
            f'{best_job["NLP Similarity"]}%'
        )

    # ==============================
    # TOP 5
    # ==============================

    st.header("📊 Top 5 Job Recommendations")

    st.dataframe(
        results_df.head(5),
        use_container_width=True
    )

    # ==============================
    # BAR CHART
    # ==============================

    st.header("📈 Job Recommendation Scores")

    chart_data = results_df.head(5).set_index(
        "Job Role"
    )["Final AI Score"]

    st.bar_chart(chart_data)

    # ==============================
    # SKILL GAP
    # ==============================

    st.header("🔍 Skill Gap Analysis")

    recommended_skills = jobs[
        best_job["Job Role"]
    ]

    missing_skills = [
        skill
        for skill in recommended_skills
        if skill not in detected_skills
    ]

    if missing_skills:

        st.warning(
            "Skills you can improve for the recommended role:"
        )

        for skill in missing_skills:
            st.write("•", skill.title())

    else:

        st.success(
            "🎉 Your resume contains all the key skills "
            "for the recommended role!"
        )

    # ==============================
    # FINAL MESSAGE
    # ==============================

    st.success(
        "✅ Resume analysis completed successfully!"
    )

else:

    st.info(
        "👆 Upload a TXT or DOCX resume to start analysis."
    )
