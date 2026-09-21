import os
import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# TITLE
# -----------------------------

st.title("🎓 Student Performance Prediction Model")

st.write(
    "Predict a student's final academic score using "
    "Machine Learning."
)


# -----------------------------
# LOAD MODEL
# -----------------------------

MODEL_PATH = "models/tuned_random_forest.joblib"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Trained model not found. "
        "Please run train_model.py first."
    )

    st.stop()


model = joblib.load(MODEL_PATH)


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("📋 Student Information")


study_hours = st.sidebar.slider(
    "Study Hours per Day",
    min_value=1.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)


attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=50,
    max_value=100,
    value=80
)


previous_score = st.sidebar.slider(
    "Previous Score",
    min_value=30,
    max_value=100,
    value=70
)


assignment_score = st.sidebar.slider(
    "Assignment Score",
    min_value=30,
    max_value=100,
    value=75
)


sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5
)


extracurricular_hours = st.sidebar.slider(
    "Extracurricular Hours",
    min_value=0,
    max_value=10,
    value=3
)


# -----------------------------
# PREDICTION
# -----------------------------

if st.button(
    "🔮 Predict Performance",
    type="primary"
):

    input_data = pd.DataFrame({

        "study_hours": [study_hours],

        "attendance": [attendance],

        "previous_score": [previous_score],

        "assignment_score": [assignment_score],

        "sleep_hours": [sleep_hours],

        "extracurricular_hours": [
            extracurricular_hours
        ]

    })


    prediction = model.predict(
        input_data
    )[0]


    st.subheader("🎯 Prediction Result")


    st.success(
        f"Predicted Final Score: {prediction:.2f} / 100"
    )


    # Performance category

    if prediction >= 75:

        st.info(
            "📈 Performance Level: High"
        )

    elif prediction >= 50:

        st.info(
            "📊 Performance Level: Moderate"
        )

    else:

        st.warning(
            "⚠️ Performance Level: Needs Improvement"
        )


# -----------------------------
# MODEL COMPARISON
# -----------------------------

st.divider()

st.header("📊 Model Comparison")


comparison_file = (
    "outputs/model_comparison.csv"
)


if os.path.exists(comparison_file):

    comparison = pd.read_csv(
        comparison_file
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

else:

    st.warning(
        "Model comparison file not found."
    )


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.header("📈 Feature Importance")


importance_file = (
    "outputs/feature_importance.csv"
)


if os.path.exists(importance_file):

    importance = pd.read_csv(
        importance_file
    )

    importance.columns = [
        "Feature",
        "Importance"
    ]

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )

else:

    st.warning(
        "Feature importance file not found."
    )


# -----------------------------
# PROJECT INFORMATION
# -----------------------------

st.divider()

st.header("ℹ️ About This Project")

st.write(
    """
This project predicts student academic performance
using Machine Learning.

The prediction uses:

• Study Hours  
• Attendance  
• Previous Score  
• Assignment Score  
• Sleep Hours  
• Extracurricular Hours  

Machine Learning models used:

• Linear Regression  
• Decision Tree  
• Random Forest  
• Tuned Random Forest  

The project also uses GridSearchCV and
5-Fold Cross-Validation for model evaluation.
"""
)