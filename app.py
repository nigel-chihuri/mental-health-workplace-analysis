# ============================================================
# STREAMLIT APP — Mental Health & Workplace Trends Analysis
# Analyst: Nigel T. Chihuri
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy.stats import chi2_contingency

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health in Tech",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background-color: #1a1d27;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2ecc71;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
    }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('OSMI_Survey_Data.csv')
    df.columns = (df.columns.str.strip().str.lower()
                    .str.replace(' ', '_', regex=False)
                    .str.replace(r'[^\w]', '_', regex=True))

    # Clean age
    df['what_is_your_age'] = pd.to_numeric(df['what_is_your_age'], errors='coerce')
    df = df[(df['what_is_your_age'] >= 16) & (df['what_is_your_age'] <= 75)]

    # Clean gender
    gender_map = {
        'male': 'Male', 'man': 'Male', 'm': 'Male', 'cis male': 'Male',
        'cis man': 'Male', 'maile': 'Male', 'mal': 'Male',
        'female': 'Female', 'woman': 'Female', 'f': 'Female',
        'cis female': 'Female', 'cis woman': 'Female',
        'femake': 'Female', 'femail': 'Female',
    }
    df['gender_clean'] = (df['what_is_your_gender'].str.strip().str.lower()
                            .map(gender_map).fillna('Non-binary / Other'))

    # Target
    target_col = 'have_you_ever_sought_treatment_for_a_mental_health_issue_from_a_mental_health_professional'
    df['sought_treatment'] = df[target_col]
    df['treatment_binary'] = df['sought_treatment'].map(
        {'Yes': 1, 'No': 0, True: 1, False: 0, 1: 1, 0: 0}
    )

    # Features
    def encode_yes_no(val):
        if pd.isnull(val): return np.nan
        v = str(val).strip().lower()
        if v in ('yes', '1', 'true'): return 1
        elif v in ('no', '0', 'false'): return 0
        return np.nan

    df['has_family_history'] = df['do_you_have_a_family_history_of_mental_illness'].apply(encode_yes_no)
    df['had_past_disorder']  = df['have_you_had_a_mental_health_disorder_in_the_past'].apply(encode_yes_no)
    df['has_diagnosis']      = df['have_you_been_diagnosed_with_a_mental_health_condition_by_a_medical_professional'].apply(encode_yes_no)
    df['is_self_employed']   = df['are_you_selfemployed'].apply(encode_yes_no)
    df['is_tech_company']    = df['is_your_employer_primarily_a_tech_companyorganization'].apply(encode_yes_no)

    remote_map = {'Always': 2, 'Sometimes': 1, 'Never': 0}
    df['works_remotely'] = df['do_you_work_remotely'].map(remote_map)

    size_map = {'1-5':1,'6-25':2,'26-100':3,'100-500':4,'500-1000':5,'More than 1000':6}
    df['company_size_ord'] = df['how_many_employees_does_your_company_or_organization_have'].map(size_map)

    disorder_map = {'Yes': 2, 'Maybe': 1, 'No': 0}
    df['current_disorder_enc'] = df['do_you_currently_have_a_mental_health_disorder'].map(disorder_map)

    bins   = [15, 24, 34, 44, 54, 75]
    labels = ['18–24', '25–34', '35–44', '45–54', '55+']
    df['age_band'] = pd.cut(df['what_is_your_age'], bins=bins, labels=labels, right=True)

    return df

df = load_data()

# ── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('models/random_forest_pipeline.pkl')
        lr = joblib.load('models/logistic_regression_pipeline.pkl')
        return rf, lr
    except:
        return None, None

rf_model, lr_model = load_models()

feature_cols = [
    'has_family_history', 'had_past_disorder', 'has_diagnosis',
    'is_tech_company', 'is_self_employed', 'works_remotely',
    'company_size_ord', 'current_disorder_enc', 'what_is_your_age',
    'gender_Male', 'gender_Female',
]

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.image("mental_health_dashboard.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Mental Health in Tech")
st.sidebar.markdown("**Analyst:** Nigel T. Chihuri")
st.sidebar.markdown("**Dataset:** OSMI Survey | 60K respondents")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 EDA Explorer",
    "📈 Statistical Findings",
    "🤖 Model Performance",
    "🧮 Predict"
])

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🧠 Mental Health & Workplace Trends")
    st.markdown("#### OSMI Mental Health in Tech Survey — Interactive Analysis")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>Total Respondents</div></div>""",
            unsafe_allow_html=True)
    with col2:
        rate = df['treatment_binary'].mean() * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{rate:.1f}%</div>
            <div class='metric-label'>Sought Treatment</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>0.987</div>
            <div class='metric-label'>Best Model AUC</div></div>""",
            unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df.shape[1]}</div>
            <div class='metric-label'>Features Analysed</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Key Findings")

    findings = [
        ("🔬", "Strongest predictor", "Prior mental health disorder history is the single strongest predictor of treatment-seeking behaviour."),
        ("👨‍👩‍👧", "Family history", "Having a family history of mental illness more than doubles the odds of seeking treatment."),
        ("⚧️", "Gender gap", "Non-binary/Other respondents seek treatment at the highest rate of any gender group."),
        ("📅", "Age peak", "Treatment-seeking peaks at age 35–44, then drops sharply at 55+ — a generational silence in the data."),
        ("🏢", "Company size", "Company size shows nearly no association with treatment-seeking — where you work matters less than what you've been through."),
    ]

    for icon, title, desc in findings:
        st.markdown(f"**{icon} {title}** — {desc}")

    st.markdown("---")
    st.image("mental_health_dashboard.png", use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.title("🔍 EDA Explorer")
    st.markdown("Filter the dataset and explore treatment-seeking patterns interactively.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender_filter = st.multiselect("Gender", df['gender_clean'].unique(),
                                        default=df['gender_clean'].unique())
    with col2:
        age_filter = st.multiselect("Age Band", df['age_band'].cat.categories.tolist(),
                                     default=df['age_band'].cat.categories.tolist())
    with col3:
        remote_filter = st.multiselect("Remote Work",
                                        df['do_you_work_remotely'].dropna().unique(),
                                        default=df['do_you_work_remotely'].dropna().unique())

    filtered = df[
        (df['gender_clean'].isin(gender_filter)) &
        (df['age_band'].isin(age_filter)) &
        (df['do_you_work_remotely'].isin(remote_filter))
    ]

    st.markdown(f"**Filtered dataset: {len(filtered):,} respondents**")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Treatment Rate by Gender")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        gt = (filtered.groupby(['gender_clean', 'treatment_binary'])
                .size().reset_index(name='count'))
        gt['pct'] = gt.groupby('gender_clean')['count'].transform(lambda x: x/x.sum()*100)
        gt['label'] = gt['treatment_binary'].map({1:'Sought Treatment', 0:'No Treatment'})
        sns.barplot(data=gt[gt['treatment_binary']==1], x='gender_clean', y='pct',
                    palette='Greens_d', ax=ax)
        ax.set_xlabel("Gender", color='white')
        ax.set_ylabel("% Sought Treatment", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig)

    with col2:
        st.markdown("#### Treatment Rate by Age Band")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        at = (filtered.groupby(['age_band', 'treatment_binary'])
                .size().reset_index(name='count'))
        at['pct'] = at.groupby('age_band')['count'].transform(lambda x: x/x.sum()*100)
        sns.barplot(data=at[at['treatment_binary']==1], x='age_band', y='pct',
                    palette='Blues_d', ax=ax)
        ax.set_xlabel("Age Band", color='white')
        ax.set_ylabel("% Sought Treatment", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Family History Impact")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        ft = (filtered.groupby(['has_family_history', 'treatment_binary'])
                .size().reset_index(name='count'))
        ft['pct'] = ft.groupby('has_family_history')['count'].transform(lambda x: x/x.sum()*100)
        ft['fam'] = ft['has_family_history'].map({1:'Yes', 0:'No'})
        sns.barplot(data=ft[ft['treatment_binary']==1], x='fam', y='pct',
                    palette='Oranges_d', ax=ax)
        ax.set_xlabel("Family History", color='white')
        ax.set_ylabel("% Sought Treatment", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig)

    with col4:
        st.markdown("#### Remote Work Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        rc = filtered['do_you_work_remotely'].value_counts()
        ax.pie(rc.values, labels=rc.index, autopct='%1.1f%%',
               colors=['#3498db','#2ecc71','#e74c3c'],
               textprops={'color':'white'})
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════
# PAGE 3 — STATISTICAL FINDINGS
# ════════════════════════════════════════════════════════════
elif page == "📈 Statistical Findings":
    st.title("📈 Statistical Findings")
    st.markdown("---")

    st.markdown("### Cramér's V — Feature Association Strength")
    st.image("cramers_v_effect_sizes.png", use_container_width=True)

    st.markdown("---")
    st.markdown("### Odds Ratio Forest Plot")
    st.image("odds_ratio_forest_plot.png", use_container_width=True)

    st.markdown("---")
    st.markdown("### Interpretation Guide")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Cramér's V Effect Sizes:**
        | V value | Strength |
        |---|---|
        | < 0.1 | Negligible |
        | 0.1 – 0.3 | Small |
        | 0.3 – 0.5 | Medium |
        | > 0.5 | Strong |
        """)
    with col2:
        st.markdown("""
        **Odds Ratios:**
        | OR value | Meaning |
        |---|---|
        | OR = 1 | No difference |
        | OR > 1 | More likely to seek treatment |
        | OR < 1 | Less likely to seek treatment |
        | CI doesn't cross 1 | Statistically significant |
        """)

# ════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>0.9867</div>
            <div class='metric-label'>Random Forest AUC</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>0.9210</div>
            <div class='metric-label'>Logistic Regression AUC</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>0.0004</div>
            <div class='metric-label'>RF CV Std Dev</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ROC Curves")
        st.image("roc_curves.png", use_container_width=True)
    with col2:
        st.markdown("### Confusion Matrices")
        st.image("confusion_matrices.png", use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Importances")
        st.image("feature_importances.png", use_container_width=True)
    with col2:
        st.markdown("### Logistic Regression Coefficients")
        st.image("lr_coefficients.png", use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ════════════════════════════════════════════════════════════
elif page == "🧮 Predict":
    st.title("🧮 Predict Treatment-Seeking Likelihood")
    st.markdown("Enter a profile below to predict the likelihood of seeking mental health treatment.")
    st.markdown("---")

    if rf_model is None:
        st.error("⚠️ Model files not found. Make sure models/ folder is present.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 16, 75, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary / Other"])
            family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
            past_disorder = st.selectbox("Had a mental health disorder in the past?", ["Yes", "No"])

        with col2:
            diagnosis = st.selectbox("Received a formal diagnosis?", ["Yes", "No"])
            current_disorder = st.selectbox("Currently have a mental health disorder?", ["Yes", "Maybe", "No"])
            remote_work = st.selectbox("Remote work?", ["Always", "Sometimes", "Never"])
            company_size = st.selectbox("Company size", ["1-5","6-25","26-100","100-500","500-1000","More than 1000"])
            is_tech = st.selectbox("Tech company?", ["Yes", "No"])
            is_self_emp = st.selectbox("Self-employed?", ["Yes", "No"])

        if st.button("🔮 Predict", use_container_width=True):
            size_map = {'1-5':1,'6-25':2,'26-100':3,'100-500':4,'500-1000':5,'More than 1000':6}
            remote_map = {'Always':2, 'Sometimes':1, 'Never':0}
            disorder_map = {'Yes':2, 'Maybe':1, 'No':0}

            input_data = pd.DataFrame([{
                'has_family_history'   : 1 if family_history == 'Yes' else 0,
                'had_past_disorder'    : 1 if past_disorder == 'Yes' else 0,
                'has_diagnosis'        : 1 if diagnosis == 'Yes' else 0,
                'is_tech_company'      : 1 if is_tech == 'Yes' else 0,
                'is_self_employed'     : 1 if is_self_emp == 'Yes' else 0,
                'works_remotely'       : remote_map[remote_work],
                'company_size_ord'     : size_map[company_size],
                'current_disorder_enc' : disorder_map[current_disorder],
                'what_is_your_age'     : age,
                'gender_Male'          : 1 if gender == 'Male' else 0,
                'gender_Female'        : 1 if gender == 'Female' else 0,
            }])

            prob = rf_model.predict_proba(input_data)[0][1]
            pred = rf_model.predict(input_data)[0]

            st.markdown("---")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                color = "#2ecc71" if prob >= 0.5 else "#e74c3c"
                st.markdown(f"""
                <div style='text-align:center; background:#1a1d27;
                            padding:30px; border-radius:15px; border:2px solid {color}'>
                    <div style='font-size:3rem; font-weight:bold; color:{color}'>{prob*100:.1f}%</div>
                    <div style='color:#aaa; font-size:1rem; margin-top:10px'>
                        Predicted likelihood of seeking treatment</div>
                    <div style='color:{color}; font-size:1.2rem; margin-top:15px; font-weight:bold'>
                        {'✅ Likely to seek treatment' if pred == 1 else '❌ Unlikely to seek treatment'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("*Prediction based on Random Forest model trained on 60,186 OSMI survey respondents. For informational purposes only.*")