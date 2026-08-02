
# --- Streamlit App: Wellness Tourism Package Purchase Predictor ---
# Loads the model trained + committed by the pipeline and serves a
# simple form UI that returns a purchase-likelihood prediction.
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.joblib"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘", layout="centered")


@st.cache_resource  # cache so the (large) model is loaded from disk only once per session
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🧘 Wellness Tourism Package -- Purchase Predictor")
st.write(
    "Enter the customer's details below to predict whether they are "
    "likely to purchase the newly launched Wellness Tourism Package."
)

model = load_model()

# Two-column form so all 18 inputs fit on one screen without excessive scrolling.
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        num_trips = st.number_input("Avg. Number of Trips per Year", min_value=0, max_value=20, value=3)
        pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    with col2:
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        passport = st.selectbox("Holds Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
        num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row dataframe matching the exact column names/order
    # the model's Pipeline was trained on, so its internal preprocessing
    # (one-hot encoding) can transform it correctly.
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

    prediction = model.predict(input_df)[0]              # 0 = won't buy, 1 = will buy
    probability = model.predict_proba(input_df)[0][1]    # predicted probability of class 1

    if prediction == 1:
        st.success(f"✅ Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase the Wellness Package (probability: {probability:.1%})")
