import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Employability Prediction System",
    page_icon="🎓",
    layout="centered"
)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf     = pickle.load(open('model/rf_model.pkl', 'rb'))
    lr     = pickle.load(open('model/lr_model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    return rf, lr, scaler

rf_model, lr_model, scaler = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Student Employability Prediction System")
st.markdown("Enter your academic profile below to predict your employability chances.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("ℹ️ About This Project")
st.sidebar.markdown("""
This ML project predicts whether a student will get placed based on academic and skill features.

**Models Used:**
- Logistic Regression
- Random Forest

**Features:**
- CGPA
- Internships
- Projects
- Aptitude Score
- Soft Skills
- Backlogs
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Disha**")

# ── Model Selection ───────────────────────────────────────────────────────────
model_choice = st.radio(
    "Choose Model:",
    ["Logistic Regression", "Random Forest"],
    horizontal=True
)

st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
    internships = st.selectbox("Number of Internships", [0, 1, 2, 3])
    projects = st.selectbox("Number of Projects", [0, 1, 2, 3, 4])

with col2:
    aptitude = st.slider("Aptitude Score (out of 100)", min_value=40, max_value=100, value=70)
    soft_skills = st.selectbox("Soft Skills Rating (1-5)", [1, 2, 3, 4, 5], index=2)
    backlogs = st.selectbox("Number of Backlogs", [0, 1, 2])

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Placement", use_container_width=True):

    input_data = np.array([[cgpa, internships, projects, aptitude, soft_skills, backlogs]])
    input_df   = pd.DataFrame(input_data,
                               columns=['CGPA','Internships','Projects',
                                        'Aptitude_Score','Soft_Skills','Backlogs'])

    if "Logistic" in model_choice:
        input_scaled = scaler.transform(input_df)
        prediction   = lr_model.predict(input_scaled)[0]
        probability  = lr_model.predict_proba(input_scaled)[0]
    else:
        prediction  = rf_model.predict(input_df)[0]
        probability = rf_model.predict_proba(input_df)[0]

    placed_prob     = probability[1] * 100
    not_placed_prob = probability[0] * 100

    st.markdown("### 📊 Prediction Result")

    if prediction == 1:
        st.success(f"✅ **LIKELY TO GET PLACED!**")
    else:
        st.error(f"❌ **NOT LIKELY TO GET PLACED**")

    col3, col4 = st.columns(2)
    col3.metric("Placement Probability",    f"{placed_prob:.1f}%")
    col4.metric("Not Placed Probability",   f"{not_placed_prob:.1f}%")

    # Tips
    st.markdown("---")
    st.markdown("### 💡 Improvement Tips")
    tips = []
    if cgpa < 7.0:
        tips.append("📚 Focus on improving your CGPA above 7.0")
    if internships == 0:
        tips.append("💼 Try to get at least 1 internship")
    if projects < 2:
        tips.append("🔨 Build more projects (aim for 2+)")
    if aptitude < 60:
        tips.append("🧠 Practice aptitude tests — score above 60")
    if soft_skills < 3:
        tips.append("🗣️ Work on communication and soft skills")
    if backlogs > 0:
        tips.append("⚠️ Clear your backlogs — they hurt placement chances")

    if tips:
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.markdown("🎉 Your profile looks great! Keep it up.")

# ── EDA Plots ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Data Insights")

tab1, tab2, tab3, tab4 = st.tabs([
    "Placement Distribution",
    "CGPA vs Placement",
    "Correlation Heatmap",
    "Feature Importance"
])

plot_files = {
    "Placement Distribution": "plots/placement_distribution.png",
    "CGPA vs Placement":      "plots/cgpa_vs_placement.png",
    "Correlation Heatmap":    "plots/correlation_heatmap.png",
    "Feature Importance":     "plots/feature_importance.png",
}

tabs = [tab1, tab2, tab3, tab4]
for tab, (title, path) in zip(tabs, plot_files.items()):
    with tab:
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Plot not found. Run train_model.py first.")
