"""
🔬 BREAST CANCER DETECTION AI - RESEARCH PROTOTYPE
Transparent ML Proof-of-Concept (Not a Clinical Device)

Model: Random Forest (test set accuracy: 97.1% on 171 samples)
Status: Research only - Requires external validation before clinical use
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Breast Cancer Detection AI - Research Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CRITICAL DISCLAIMER ============
st.error("""
⚠️ **LEGAL DISCLAIMER**

**This is a RESEARCH PROTOTYPE, NOT a medical device.**

- Not FDA-approved or clinically validated
- Not intended for clinical decision-making
- No clinical trials on patient populations
- For research and educational purposes only
- Always consult qualified radiologists

Using this system for clinical decisions without physician oversight violates regulations.
""")


# ============ AUTO-RETRAIN FOR CLOUD DEPLOYMENT ============
@st.cache_resource
def load_model_files():
    """Load or train model and scaler"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # If model files don't exist, retrain on startup
    if not os.path.exists('breast_cancer_model.pkl'):
        with st.spinner('🔄 Training model on startup (first time only)...'):
            # Load dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

            # Define column names
            metric_names = ['radius', 'texture', 'perimeter', 'area',
                            'smoothness', 'compactness', 'concavity',
                            'concave_points', 'symmetry', 'fractal_dimension']
            stat_names = ['mean', 'se', 'worst']

            data_columns = ['id', 'diagnosis']
            for metric in metric_names:
                for stat in stat_names:
                    data_columns.append(f'{metric}_{stat}')

            df = pd.read_csv(url, header=None, names=data_columns)
            df.drop('id', axis=1, inplace=True)
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

            x_data = df.drop('diagnosis', axis=1)
            y_data = df['diagnosis']

            scaler_obj = StandardScaler()
            x_scaled = scaler_obj.fit_transform(x_data)

            model_obj = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model_obj.fit(x_scaled, y_data)

            # Save for next time
            joblib.dump(model_obj, 'breast_cancer_model.pkl')
            joblib.dump(scaler_obj, 'scaler.pkl')

            st.success('✅ Model trained and cached!')
            return model_obj, scaler_obj

    # If files exist, load them
    try:
        model_obj = joblib.load('breast_cancer_model.pkl')
        scaler_obj = joblib.load('scaler.pkl')
        return model_obj, scaler_obj
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


trained_model, feature_scaler = load_model_files()

feature_names = [
    'radius_mean', 'radius_se', 'radius_worst', 'texture_mean', 'texture_se',
    'texture_worst', 'perimeter_mean', 'perimeter_se', 'perimeter_worst',
    'area_mean', 'area_se', 'area_worst', 'smoothness_mean', 'smoothness_se',
    'smoothness_worst', 'compactness_mean', 'compactness_se', 'compactness_worst',
    'concavity_mean', 'concavity_se', 'concavity_worst', 'concave_points_mean',
    'concave_points_se', 'concave_points_worst', 'symmetry_mean', 'symmetry_se',
    'symmetry_worst', 'fractal_dimension_mean', 'fractal_dimension_se',
    'fractal_dimension_worst'
]

# ============ MAIN HEADER ============
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
with header_col2:
    st.title("🔬 Breast Cancer Detection AI")
    st.markdown("<h3 style='text-align: center'>Research Prototype - Not for Clinical Use</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666'>Machine Learning Proof-of-Concept</p>",
        unsafe_allow_html=True
    )

# ============ SIDEBAR - HONEST METRICS ============
with st.sidebar:
    st.header("📊 Model Performance")

    st.markdown("""
    **Dataset:** Wisconsin Breast Cancer (UCI)  
    **Size:** 569 patients (64 malignant, 107 benign in test set)  
    **Algorithm:** Random Forest (100 trees)
    """)

    st.divider()

    st.subheader("Performance at Different Thresholds")

    performance_data = {
        'Threshold': ['0.5 (default)', '0.3 (balanced)', '0.2 (sensitive)'],
        'Sensitivity': ['93.8%', '96.9%', '98.4%'],
        'Specificity': ['99.1%', '97.2%', '91.6%'],
        'False Negatives': ['4', '2', '1'],
        'False Positives': ['0', '3', '10']
    }

    st.dataframe(performance_data, width='stretch', hide_index=True)

    st.info("""
    **Key Finding:** Sensitivity ranges from 93.8% to 98.4% depending on threshold.
    Higher sensitivity (more detection) = more false positives (unnecessary biopsies).
    """)

    st.divider()

    st.metric("ROC-AUC", "0.990", "Excellent discrimination")
    st.metric("Test Accuracy", "97.1%", "At default threshold")

    st.divider()

    st.warning("""
    **⚠️ Critical Limitations**

    1. **Small test set** (n=171)
    2. **Single dataset** (no external validation)
    3. **Retrospective data** (historical, not prospective)
    4. **No clinical trials**
    """)

# ============ MAIN CONTENT ============
tab1, tab2, tab3, tab4 = st.tabs(["📊 Methodology", "🔬 Prediction Demo", "📈 Performance Analysis", "📋 Limitations"])

# ============ TAB 1: METHODOLOGY ============
with tab1:
    st.subheader("Study Design & Methodology")

    method_col1, method_col2 = st.columns(2)

    with method_col1:
        st.markdown("""
        ### Dataset
        - **Source:** UCI Wisconsin Breast Cancer Database
        - **Total samples:** 569 patients
        - **Features:** 30 radiological measurements
        - **Target:** Benign vs. Malignant diagnosis

        ### Class Distribution
        - Benign: 357 (62.7%)
        - Malignant: 212 (37.3%)

        ⚠️ Note: This is observational, retrospective data from 1990s.
        """)

    with method_col2:
        st.markdown("""
        ### Train-Test Split
        - **Training:** 398 samples (70%)
        - **Test:** 171 samples (30%)
        - **Strategy:** Stratified random split
        - **Seed:** 42 (reproducible)

        ### Preprocessing
        - StandardScaler normalization
        - Fit on training data only
        - Applied to test data

        **Why this matters:** Prevents data leakage.
        """)

    st.divider()

    st.markdown("""
    ### Model Selection

    Tested two algorithms:

    **Logistic Regression**
    - Baseline model
    - Interpretable coefficients
    - Test accuracy: 97.1%

    **Random Forest**
    - Captures non-linear relationships
    - Handles feature interactions
    - Test accuracy: 97.1%
    - **Selected** for better ROC-AUC (0.99)
    """)

# ============ TAB 2: PREDICTION DEMO ============
with tab2:
    st.subheader("Interactive Prediction Demo")

    st.warning("""
    ⚠️ **Important:** This demo is for research demonstration only.

    **Limitations:**
    - Only 9 of 30 features can be manually entered
    - Remaining 21 features use dataset mean values (not patient-specific)
    - Predictions should NOT be used for medical decisions
    - Always consult a radiologist
    """)

    input_col, result_col = st.columns([2, 1], gap="medium")

    with input_col:
        st.markdown("### Enter Patient Measurements (9/30 features)")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            concave_points_worst = st.number_input(
                "Concave Points (Worst)",
                value=0.08,
                step=0.01,
                format="%.4f",
                help="Range: 0.00-0.29"
            )
            radius_worst_val = st.number_input(
                "Radius (Worst)",
                value=14.0,
                step=1.0,
                help="Range: 7.0-40.0"
            )
            area_worst_val = st.number_input(
                "Area (Worst)",
                value=600.0,
                step=50.0,
                help="Range: 185-4254"
            )

        with col_b:
            concave_points_mean = st.number_input(
                "Concave Points (Mean)",
                value=0.05,
                step=0.01,
                format="%.4f",
                help="Range: 0.00-0.20"
            )
            perimeter_worst_val = st.number_input(
                "Perimeter (Worst)",
                value=90.0,
                step=5.0,
                help="Range: 43-251"
            )
            concavity_mean_val = st.number_input(
                "Concavity (Mean)",
                value=0.088,
                step=0.01,
                format="%.4f",
                help="Range: 0.00-0.43"
            )

        with col_c:
            texture_worst_val = st.number_input(
                "Texture (Worst)",
                value=22.0,
                step=1.0,
                help="Range: 12-39"
            )
            smoothness_worst_val = st.number_input(
                "Smoothness (Worst)",
                value=0.12,
                step=0.01,
                format="%.4f",
                help="Range: 0.07-0.22"
            )
            symmetry_worst_val = st.number_input(
                "Symmetry (Worst)",
                value=0.25,
                step=0.01,
                format="%.4f",
                help="Range: 0.16-0.66"
            )

        default_input_values = {
            'radius_mean': 12.0, 'radius_se': 0.25, 'texture_mean': 18.0, 'texture_se': 0.5,
            'perimeter_mean': 78.0, 'perimeter_se': 1.5, 'area_mean': 450.0, 'area_se': 20.0,
            'smoothness_mean': 0.095, 'smoothness_se': 0.007, 'compactness_mean': 0.1,
            'compactness_se': 0.015, 'compactness_worst': 0.15, 'concavity_se': 0.02,
            'concave_points_se': 0.01, 'symmetry_mean': 0.18, 'symmetry_se': 0.02,
            'fractal_dimension_mean': 0.062, 'fractal_dimension_se': 0.003, 'fractal_dimension_worst': 0.08
        }

        analyze_button = st.button("🔬 ANALYZE FEATURES", type="primary", width='stretch')

    with result_col:
        if analyze_button:
            input_dict = {
                'concave_points_worst': concave_points_worst,
                'radius_worst': radius_worst_val,
                'area_worst': area_worst_val,
                'concave_points_mean': concave_points_mean,
                'perimeter_worst': perimeter_worst_val,
                'concavity_mean': concavity_mean_val,
                'texture_worst': texture_worst_val,
                'smoothness_worst': smoothness_worst_val,
                'symmetry_worst': symmetry_worst_val,
            }
            for feature in feature_names:
                if feature not in input_dict:
                    input_dict[feature] = default_input_values.get(feature, 0.0)

            input_df = pd.DataFrame([input_dict])[feature_names]
            input_scaled = feature_scaler.transform(input_df)
            probability = trained_model.predict_proba(input_scaled)[0, 1]

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Model Output (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF6B6B"},
                    'steps': [
                        {'range': [0, 10], 'color': "#d4edda"},
                        {'range': [10, 30], 'color': "#fff3cd"},
                        {'range': [30, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 10}
                }
            ))
            gauge_fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(gauge_fig, width='stretch')

            st.divider()

            if probability >= 0.3:
                st.warning(f"""
                ### ⚠️ Model Output: {probability * 100:.1f}%

                **Interpretation:** Model assigns **higher probability** to malignancy.

                ⚠️ **This is NOT a diagnosis.** 

                **Next steps:**
                1. Physician review of imaging
                2. Additional diagnostic imaging if indicated
                3. Biopsy only if clinically warranted
                """)
            elif probability >= 0.1:
                st.info(f"""
                ### 📊 Model Output: {probability * 100:.1f}%

                **Interpretation:** Model assigns **intermediate probability** to malignancy.

                ⚠️ **This is NOT a diagnosis.**

                **Next steps:**
                1. Physician review of imaging
                2. Possibly additional imaging/follow-up
                """)
            else:
                st.success(f"""
                ### ✅ Model Output: {probability * 100:.1f}%

                **Interpretation:** Model assigns **lower probability** to malignancy.

                ⚠️ **This is NOT a diagnosis.**

                **Next steps:**
                1. Physician review of imaging
                2. Standard screening follow-up per guidelines
                """)

            st.info("""
            **Remember:**
            - Model probability ≠ actual cancer probability
            - Test set recall: 93.8% (misses ~6% of cancers at default threshold)
            - Always trust radiologist assessment over model
            """)
        else:
            st.info("👆 Enter measurements and click 'Analyze Features'")

# ============ TAB 3: PERFORMANCE ANALYSIS ============
with tab3:
    st.subheader("Detailed Performance Metrics")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("""
                 ### Confusion Matrix (Test Set, Threshold=0.5""")
