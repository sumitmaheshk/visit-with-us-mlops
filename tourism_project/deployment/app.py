"""
Streamlit App - Wellness Tourism Package Purchase Predictor
--------------------------------------------------------------
Loads the trained pipeline (preprocessing + XGBoost model) that the
GitHub Actions pipeline committed to this folder, collects a customer's
details through a simple form, packages them into a single-row dataframe
with the exact column names/order the model was trained on, and displays
the purchase prediction with its probability.
"""

import joblib
import pandas as pd
import streamlit as st

# Streamlit Community Cloud's working directory is the repository root,
# so this path is relative to the repo root, not to this file.
MODEL_PATH = "tourism_project/deployment/best_model.joblib"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained pipeline once per session and cache it."""
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🧘 Wellness Tourism Package — Purchase Predictor")
st.write(
    "This tool estimates whether a customer is likely to purchase the newly "
    "launched **Wellness Tourism Package**, based on their profile and how "
    "they responded to the sales pitch. Enter the customer's details below."
)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Profile")
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthly_income = st.number_input(
            "Monthly Income", min_value=1000, max_value=100000, value=20000, step=500
        )
        passport = st.selectbox("Holds Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

    with col2:
        st.subheader("Trip & Pitch Details")
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
        num_trips = st.number_input("Avg. Number of Trips per Year", min_value=0, max_value=20, value=3)
        preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0], index=0)
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox(
            "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        duration_of_pitch = st.number_input(
            "Duration of Pitch (minutes)", min_value=1, max_value=60, value=10
        )
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Column names/order must match what the model was trained on
    # (see CATEGORICAL + NUMERIC in tourism_project/model_building/train.py).
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase the Wellness Package (probability: {probability:.1%})")

    st.progress(min(max(probability, 0.0), 1.0))
    with st.expander("Show input data sent to the model"):
        st.dataframe(input_df)
