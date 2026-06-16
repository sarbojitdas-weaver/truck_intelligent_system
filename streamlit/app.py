import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sys

# ==============================================================================
# 1. BLUEPRINT NAMESPACE RESCUE (PREVENTS DESERIALIZATION ERRS)
# ==============================================================================
class VehicleRecommendationEngine:
    def __init__(self, price_model, shap_explainer, feature_lookup, model_popularity):
        self.price_model = price_model
        self.shap_explainer = shap_explainer
        self.feature_lookup = feature_lookup
        self.model_popularity = model_popularity

    def predict_price(self, make, model, age, mileage):
        row = self.feature_lookup[
            (self.feature_lookup["make"] == make) &
            (self.feature_lookup["model"] == model)
        ]
        if len(row) == 0:
            row = pd.DataFrame([{
                "body": "Sedan", "transmission_type": "Automatic", "seller": "Dealer",
                "source_type": "retail", "truck_type": "Standard", "status": "active",
                "mmr": self.feature_lookup["mmr"].median(),
                "days_on_market": self.feature_lookup["days_on_market"].median()
            }])
        mileage_per_year = mileage / (age + 1)
        input_df = pd.DataFrame([{
            "make": make, "model": model, "body": row["body"].values[0],
            "transmission_type": row["transmission_type"].values[0], "seller": row["seller"].values[0],
            "source_type": row["source_type"].values[0], "truck_type": row["truck_type"].values[0],
            "status": row["status"].values[0], "age": age, "mileage": mileage,
            "mmr": row["mmr"].values[0], "days_on_market": row["days_on_market"].values[0],
            "mileage_per_year": mileage_per_year
        }])
        return float(self.price_model.predict(input_df)[0]), input_df

    def explain_price(self, input_df):
        try:
            shap_values = self.shap_explainer.shap_values(input_df)
            if isinstance(shap_values, list): shap_values = shap_values[0]
            shap_values = np.array(shap_values).flatten()
            return pd.DataFrame({"feature": list(input_df.columns), "impact": shap_values[:len(input_df.columns)]})
        except:
            return pd.DataFrame()

    def popularity_score(self, make, model):
        row = self.model_popularity[(self.model_popularity["make"] == make) & (self.model_popularity["model"] == model)]
        return 0.50 if len(row) == 0 else row["popularity_score"].values[0]

    def mileage_health_score(self, age, mileage):
        expected = max(age * 12000, 1)
        ratio = mileage / expected
        if ratio <= 0.80: return 1.0
        elif ratio <= 1.00: return 0.80
        elif ratio <= 1.20: return 0.60
        elif ratio <= 1.40: return 0.30
        else: return 0.10

    def build_human_reason(self, raw_gap, mileage_score, popularity, recommendation):
        reason = []
        if raw_gap > 0.20: reason.append("This car is priced below market value, making it a good deal.")
        elif raw_gap < -0.10: reason.append("This car is priced above market value, indicating overpricing.")
        else: reason.append("This car is fairly priced based on market comparison.")
        if mileage_score > 0.80: reason.append("It has low mileage for its age, which is a positive sign.")
        elif mileage_score < 0.30: reason.append("It has high mileage for its age, which is a concern.")
        if popularity > 0.60: reason.append("This model is in strong market demand.")
        if recommendation in ["STRONG BUY", "BUY"]: reason.append("Overall conditions suggest it is a good buying opportunity.")
        elif recommendation in ["AVOID", "WEAK AVOID"]: reason.append("Overall signals suggest avoiding this purchase.")
        return " ".join(reason)

    def recommend(self, make, model, age, mileage, listing_price):
        predicted_price, input_df = self.predict_price(make, model, age, mileage)
        shap_df = self.explain_price(input_df)
        popularity = self.popularity_score(make, model)
        mileage_score = self.mileage_health_score(age, mileage)
        raw_gap = (predicted_price - listing_price) / predicted_price
        discount_score = np.tanh(2 * raw_gap)
        final_score = (0.70 * discount_score + 0.20 * mileage_score + 0.10 * popularity)
        if final_score >= 0.70: recommendation = "STRONG BUY"
        elif final_score >= 0.55: recommendation = "BUY"
        elif final_score >= 0.40: recommendation = "HOLD"
        elif final_score >= 0.25: recommendation = "WEAK AVOID"
        else: recommendation = "AVOID"
        reason = self.build_human_reason(raw_gap, mileage_score, popularity, recommendation)
        return {"make": make, "model": model, "recommendation": recommendation, "predicted_market_price": round(predicted_price, 2), "listing_price": listing_price, "score": round(final_score, 3), "reason": reason}

# Inject custom model class definition directly into top-level execution scope
sys.modules['__main__'].VehicleRecommendationEngine = VehicleRecommendationEngine

# ==============================================================================
# 2. STREAMLIT CONFIGURATION & ENGINES DE-SERIALIZATION
# ==============================================================================
st.set_page_config(page_title="Valuation Engine Dashboard", page_icon="🚘", layout="centered")

HF_REPO_ID = "sarbojit-weavers/recommendation_model"
MODEL_FILENAME = "vehicle_recommendation_engine.pkl"

# Initialize persistence flags to track click states across user interactions
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

@st.cache_resource
def load_ml_engine():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)
        except: pass
    return None

engine_instance = load_ml_engine()

# ==============================================================================
# 3. STATLESS HARDCODED MAKE/MODEL MAPPING (NO DATASET FILE DEPENDENCY)
# ==============================================================================
MAKE_MODEL_MAP = {
    "BMW": ["3 Series"],
    "Chevrolet": ["Impala"],
    "Ford": ["F-150", "Fusion"],
    "Honda": ["Accord", "Civic"],
    "Hyundai": ["Elantra"],
    "Nissan": ["Altima"],
    "Toyota": ["Camry", "Corolla"]
}

st.title("🎯 Underwriting & Vehicle Evaluation Dashboard")
st.markdown("Select an asset context profile layout below to test decision workflows.")

col1, col2 = st.columns(2)
with col1:
    selected_make = st.selectbox("🗂️ Step 1: Select Brand Manufacturer", options=sorted(list(MAKE_MODEL_MAP.keys())))

with col2:
    selected_model = st.selectbox("🚘 Step 2: Select Sub-Derivative Series", options=sorted(MAKE_MODEL_MAP[selected_make]))

# Track whether selection parameters changed; reset evaluation state if fields shift
if "last_selection" not in st.session_state:
    st.session_state.last_selection = (selected_make, selected_model)

if st.session_state.last_selection != (selected_make, selected_model):
    st.session_state.evaluated = False
    st.session_state.last_selection = (selected_make, selected_model)

# ==============================================================================
# 4. VARIABLE APPRAISAL FORM INPUT SEGMENT
# ==============================================================================
st.markdown("---")
st.subheader("🛠️ Vehicle Physical Properties Configuration")

with st.form("appraisal_form"):
    c1, c2 = st.columns(2)
    with c1:
        inp_age = st.number_input("Vehicle Relative Age (Years)", min_value=0, max_value=30, value=5)
        inp_mileage = st.number_input("Odometer Overage Mileage Status (mi)", min_value=100, max_value=600000, value=50000)
    with c2:
        inp_mmr = st.number_input("Base Reference Market MMR Value ($)", min_value=100.0, value=15000.0)
        inp_price = st.number_input("Target Sourcing Listing Cost ($)", min_value=100.0, value=14000.0)
        
    submit = st.form_submit_button("Run Underwriting Risk Optimization Appraisal", use_container_width=True)

# Update state metrics if form action occurs
if submit:
    st.session_state.evaluated = True

# ==============================================================================
# 5. METRICS REPORTING LAYER (ONLY REVEALED ON SELECTION TRIGGER CLICK)
# ==============================================================================
if st.session_state.evaluated:
    if engine_instance is not None:
        try:
            out = engine_instance.recommend(selected_make, selected_model, inp_age, inp_mileage, inp_price)
            verdict, fair_price, score, explanation = out["recommendation"], out["predicted_market_price"], out["score"], out["reason"]
        except Exception as e:
            # Mathematical evaluation backup configuration block
            ratio = inp_mileage / max(inp_age * 12000, 1)
            m_score = 1.0 if ratio <= 0.80 else (0.80 if ratio <= 1.00 else (0.60 if ratio <= 1.20 else (0.30 if ratio <= 1.40 else 0.10)))
            r_gap = (inp_mmr - inp_price) / inp_mmr
            score = round(0.70 * np.tanh(2 * r_gap) + 0.20 * m_score + 0.10 * 0.60, 3)
            verdict = "STRONG BUY" if score >= 0.70 else ("BUY" if score >= 0.55 else ("HOLD" if score >= 0.40 else ("WEAK AVOID" if score >= 0.25 else "AVOID")))
            explanation = f"Evaluated based on fallback index scoring metrics with variance of {r_gap:+.2f}%."
            fair_price = inp_mmr
    else:
        # Static baseline logic backup if model object fails to unpickle
        ratio = inp_mileage / max(inp_age * 12000, 1)
        m_score = 1.0 if ratio <= 0.80 else (0.80 if ratio <= 1.00 else (0.60 if ratio <= 1.20 else (0.30 if ratio <= 1.40 else 0.10)))
        r_gap = (inp_mmr - inp_price) / inp_mmr
        score = round(0.70 * np.tanh(2 * r_gap) + 0.20 * m_score + 0.10 * 0.50, 3)
        verdict = "STRONG BUY" if score >= 0.70 else ("BUY" if score >= 0.55 else ("HOLD" if score >= 0.40 else ("WEAK AVOID" if score >= 0.25 else "AVOID")))
        
        reasons = []
        if r_gap > 0.20: reasons.append("This car is priced below market value, making it a good deal.")
        elif r_gap < -0.10: reasons.append("This car is priced above market value, indicating overpricing.")
        else: reasons.append("This car is fairly priced based on market comparison.")
        if m_score > 0.80: reasons.append("It has low mileage for its age, which is a positive sign.")
        elif m_score < 0.30: reasons.append("It has high mileage for its age, which is a concern.")
        reasons.append("Overall conditions suggest it is a good buying opportunity." if verdict in ["STRONG BUY", "BUY"] else "Overall signals suggest avoiding this purchase.")
        explanation = " ".join(reasons)
        fair_price = inp_mmr

    # Graphical Metrics Presentation Structure
    st.markdown("### 📊 Sourcing Performance Analytics")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Predicted Market Target Fair Price", f"${fair_price:,.2f}")
    rc2.metric("Procurement Cost Advantage Delta", f"{((fair_price - inp_price) / fair_price) * 100:+.2f}%")
    rc3.metric("Composite Decision Metric Score", f"{score:.3f}")

    # Display stylized output recommendations matching evaluation parameters
    if verdict == "STRONG BUY":
        st.success(f"🏆 **UNDERWRITING DECISION VERDICT:** **{verdict}** \n\n {explanation}")
    elif verdict == "BUY":
        st.info(f"💎 **UNDERWRITING DECISION VERDICT:** **{verdict}** \n\n {explanation}")
    elif verdict == "HOLD":
        st.warning(f"⚖️ **UNDERWRITING DECISION VERDICT:** **{verdict}** \n\n {explanation}")
    elif verdict in ["WEAK AVOID", "AVOID"]:
        st.error(f"❌ **UNDERWRITING DECISION VERDICT:** **{verdict}** \n\n {explanation}")
else:
    # Quiet layout placeholders showing until evaluated
    st.info("💡 Adjust the metrics above and trigger the appraisal action button to display recommendation metrics.")