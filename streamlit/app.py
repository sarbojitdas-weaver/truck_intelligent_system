import pickle
import pandas as pd
import streamlit as st
from catboost import Pool
from huggingface_hub import hf_hub_download

# Configure a clean wide layout
st.set_page_config(
    page_title="Truck Price Intelligent Evaluator", page_icon="🚚", layout="wide"
)

# --- UPDATED HUGGING FACE DOWNLOAD LOGIC ---
@st.cache_resource
def load_prediction_model():
    """Downloads the trained pickle model framework from Hugging Face Hub 
    natively and loads it directly into the execution container memory space.
    """
    # Replace 'username/repo-name' with your actual Hugging Face Repository identifier
    # Replace 'recommendation_model.pkl' with the exact filename inside your HF repo
    model_path = hf_hub_download(
        repo_id="sarbojit-weavers/recommendation_model", 
        filename="recommendation_model.pkl"
    )
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_prediction_model()
except Exception as e:
    st.error(
        f"🚨 **Failed to download or unpack model from Hugging Face Hub!** \n\n"
        f"**Error Details:** `{str(e)}`"
    )
    st.stop()

# Dataset mapping for dependent dropdowns
MAKE_MODEL_MAPPING = {
    'Freightliner': ['Cascadia', 'Columbia', 'Coronado'], 
    'International': ['LT625', 'ProStar'], 
    'Kenworth': ['W900', 'T680', 'T880'], 
    'Mack': ['Anthem'], 
    'Peterbilt': ['579', '389', '567'], 
    'Volvo': ['VNL860', 'VNL760']
}

# English translations for your feature names
HUMAN_FEATURE_NAMES = {
    "source_type": "listing source type",
    "year": "year of manufacture",
    "make": "manufacturer branding",
    "model": "model variant",
    "mileage": "mileage reading",
    "location": "geographic location",
    "engine": "engine configuration",
    "axles": "axle setup",
    "transmission_type": "transmission type",
    "transmission_speed": "gear speed count",
    "apu": "APU status",
    "truck_type": "chassis cab style",
    "sleeper_size": "sleeper compartment size",
    "overhaul_paperwork": "overhaul paperwork validation",
    "wheelbase_category": "wheelbase frame layout"
}

# Categorical drop-down selection options
SOURCE_TYPES = ["retail", "dealer", "auction"]
LOCATIONS = ['Atlanta, GA', 'Phoenix, AZ', 'Houston, TX', 'Fort Worth, TX', 'Dallas, TX', 'Chicago, IL', 'Salt Lake City, UT', 'Indianapolis, IN', 'Memphis, TN', 'Kansas City, MO']
ENGINES = ["Detroit DD15", "Cummins X15", "Volvo D13", "Detroit Series 60", "PACCAR MX-13"]
AXLE_OPTIONS = ["3", "2", "4"]
TRANSMISSION_TYPES = ["Automatic", "Manual", "AMT"]
TRUCK_TYPES = ["Sleeper", "Day Cab"]
SLEEPER_SIZES = ["72", "60", "80", "Missing"]
WHEELBASE_CATEGORIES = ["Medium Wheelbase", "Short Wheelbase", "Long Wheelbase"]

categorical_features = [
    "source_type", "make", "model", "location", "engine", 
    "axles", "transmission_type", "truck_type", "sleeper_size", "wheelbase_category"
]

# --- UI HEADER ---
st.title("🚚 Truck Price Smart Evaluator")
st.write("Configure specifications from an external listing to run an automated market comparison analysis.")
st.markdown("---")

# --- TWO-COLUMN LAYOUT ---
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 Select Deal Details")

    listing_price = st.number_input(
        "Website Asking Price ($)", min_value=1000, max_value=500000, value=62000, step=500
    )
    mileage = st.number_input(
        "Odometer Mileage Reading", min_value=0, max_value=2000000, value=520000, step=10000
    )
    year = st.selectbox("Year of Manufacture", list(range(2027, 1999, -1)), index=12)

    selected_make = st.selectbox("Manufacturer (Make)", list(MAKE_MODEL_MAPPING.keys()))
    available_models = MAKE_MODEL_MAPPING[selected_make]
    selected_model = st.selectbox("Model Name", available_models)

    source_type = st.selectbox("Listing Source Type", SOURCE_TYPES)
    location = st.selectbox("Location State Code", LOCATIONS)
    engine = st.selectbox("Engine Variant", ENGINES)
    axles = st.selectbox("Axle Count", AXLE_OPTIONS)
    transmission_type = st.selectbox("Transmission Type", TRANSMISSION_TYPES)

    transmission_speed = st.number_input(
        "Transmission Gears (Speed)", min_value=1, max_value=18, value=12, step=1
    )

    apu = st.selectbox("Is an APU Installed?", ["No", "Yes"])
    truck_type = st.selectbox("Truck Chassis Type", TRUCK_TYPES)
    sleeper_size = st.selectbox("Sleeper Size (Inches)", SLEEPER_SIZES)
    overhaul_paperwork = st.selectbox("Overhaul Paperwork Available?", ["No", "Yes"])
    wheelbase_category = st.selectbox("Wheelbase Category", WHEELBASE_CATEGORIES)

    st.markdown(" ")
    evaluate_clicked = st.button(
        "📊 Evaluate Deal", type="primary", use_container_width=True
    )

# --- RESULTS AREA ---
with col_result:
    st.subheader("📈 Evaluation Report")

    if evaluate_clicked:
        apu_int = 1 if apu == "Yes" else 0
        overhaul_int = 1 if overhaul_paperwork == "Yes" else 0

        input_dict = {
            "source_type": source_type, "year": int(year), "make": selected_make, "model": selected_model,
            "mileage": int(mileage), "location": location, "engine": engine, "axles": str(axles),
            "transmission_type": transmission_type, "transmission_speed": int(transmission_speed),
            "apu": apu_int, "truck_type": truck_type, "sleeper_size": sleeper_size,
            "overhaul_paperwork": overhaul_int, "wheelbase_category": wheelbase_category,
        }

        features_ordered = [
            "source_type", "year", "make", "model", "mileage", "location", "engine", "axles",
            "transmission_type", "transmission_speed", "apu", "truck_type", "sleeper_size",
            "overhaul_paperwork", "wheelbase_category",
        ]

        input_df = pd.DataFrame([input_dict])[features_ordered]

        for col in input_df.select_dtypes(include=["object", "category", "string"]).columns:
            input_df[col] = input_df[col].fillna("Missing").astype(str)

        # Predictions
        predicted_fmv = model.predict(input_df)[0]
        deviation = (listing_price - predicted_fmv) / predicted_fmv
        price_difference = abs(listing_price - predicted_fmv)

        # Display Standard Recommendation Badge
        if deviation <= -0.10:
            st.success(f"### Recommendation: **🟢 BUY**\nThis truck is highly underpriced by {abs(deviation)*100:.1f}% compared to fair market value!")
        elif deviation >= 0.10:
            st.error(f"### Recommendation: **🔴 AVOID**\nThis truck is overpriced by {deviation*100:.1f}% compared to fair market value.")
        else:
            st.warning(f"### Recommendation: **🟡 HOLD / NEGOTIATE**\nThe listing price matches expected fair market variance (+/-10%).")

        st.markdown("---")

        # Scorecard Metrics
        m1, m2 = st.columns(2)
        m1.metric(label="Predicted Fair Market Price", value=f"${predicted_fmv:,.2f}")
        m2.metric(
            label="Market Pricing Variance",
            value=f"{deviation * 100:+.2f}%",
            delta=f"${price_difference:,.2f} {'Overpriced' if deviation > 0 else 'Underpriced'}",
            delta_color="inverse",
        )

        # --- NATIVE SHAP VALUE GENERATION ---
        input_pool = Pool(data=input_df, cat_features=categorical_features)
        shap_values = model.get_feature_importance(data=input_pool, type="ShapValues")[0]

        feature_impacts = dict(zip(features_ordered, shap_values[:-1]))
        sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        pos_drivers = [(feat, val) for feat, val in sorted_impacts if val > 100][:2]
        neg_drivers = [(feat, val) for feat, val in sorted_impacts if val < -100][:2]

       # --- DYNAMIC REASONING ENGINE ACCORDING TO BUY/HOLD/AVOID STATUS ---
        st.markdown("---")
        st.markdown("### 🔍 Market Evaluation Summary")

        # Create clean text components for the top features
        top_pos_name = HUMAN_FEATURE_NAMES.get(pos_drivers[0][0], pos_drivers[0][0]) if pos_drivers else "general specs"
        top_pos_val = f"${pos_drivers[0][1]:,.2f}" if pos_drivers else "$0.00"

        top_neg_name = HUMAN_FEATURE_NAMES.get(neg_drivers[0][0], neg_drivers[0][0]) if neg_drivers else "normal wear"
        top_neg_val = f"${abs(neg_drivers[0][1]):,.2f}" if neg_drivers else "$0.00"

        if deviation <= -0.10:
            # 🟢 BUY LOGIC EXECUTIVE TEXT
            st.write(
                f"Our intelligent system has analyzed this truck's unique configuration against active market data points. "
                f"This asset represents a highly lucrative opportunity because it is listed at a deep **{abs(deviation)*100:.1f}% market discount**. "
                f"The seller is leaving **${price_difference:,.2f}** on the table, allowing you to secure instant equity immediately upon purchase."
            )
            
            st.markdown("#### 🚀 Why is this a strong investment choice?")
            if pos_drivers:
                for feat, val in pos_drivers:
                    readable = HUMAN_FEATURE_NAMES.get(feat, feat)
                    if feat == "mileage":
                        st.write(f"✔ **Premium Mileage Status:** This unit features low usage parameters for its lifecycle tier, injecting a major **+${val:,.2f}** premium back into true valuation books.")
                    elif feat == "sleeper_size":
                        st.write(f"✔ **Premium Cabin Configuration:** The highly desirable sleeper layout minimizes driver turnover friction, pushing baseline buyer valuation up by **+${val:,.2f}**.")
                    else:
                        st.write(f"✔ The truck's **{readable}** shows outstanding market configuration strength, generating a cash-equivalent advantage of **+${val:,.2f}**.")

            st.markdown(" ")
            # Spaced out and clean markdown formatting block
            st.info(
                f"💡 **Final Action Plan: BUY IMMEDIATELY**\n\n"
                f"• **Major Premium Advantage:** The truck's **{top_pos_name}** adds an exceptional asset premium of **{top_pos_val}**.\n\n"
                f"• **Minor Market Penalty:** Normal baseline depreciation from its **{top_neg_name}** cuts away **-{top_neg_val}** from top-tier pricing.\n\n"
                f"**Verdict:** The premium benefits heavily outweigh the aging depreciation factors. Move fast before a competitor locks this asset down."
            )

        elif deviation >= 0.10:
            # 🔴 AVOID LOGIC EXECUTIVE TEXT
            st.write(
                f"Our intelligent system has completed the structural analysis of this vehicle listing. "
                f"This transaction is flagged as **highly unfavorable** because the seller's asking price is **{deviation*100:.1f}% over fair market valuation**. "
                f"Purchasing this vehicle at the requested rate introduces an immediate financial risk of **${price_difference:,.2f}** in negative equity."
            )
            
            st.markdown("#### ⚠️ Primary Financial Risk Hazards")
            if neg_drivers:
                for feat, val in neg_drivers:
                    readable = HUMAN_FEATURE_NAMES.get(feat, feat)
                    if feat == "year":
                        st.write(f"❌ **Aggressive Time Depreciation:** This truck's manufacture year creates heavy, unrecoverable lifecycle degradation, forcing a value penalty of **-${abs(val):,.2f}**.")
                    elif feat == "apu":
                        st.write(f"❌ **Deficient APU Outfitting:** The absent or poorly suited Auxiliary Power configuration fails to meet fleet idle constraints, triggering a **-${abs(val):,.2f}** market adjustment.")
                    else:
                        st.write(f"❌ The configuration choices on **{readable}** act as a severe market weight, dropping baseline buyer value parameters by **-${abs(val):,.2f}**.")

            st.markdown(" ")
            st.error(
                f"💡 **Final Action Plan: AVOID DEAL**\n\n"
                f"• **Major Pricing Risk:** The truck's valuation is heavily anchored downward by its **{top_neg_name}** (**-{top_neg_val}**).\n\n"
                f"• **Inadequate Offsets:** Minor structural advantages from its **{top_pos_name}** (**{top_pos_val}**) are simply not enough to cover the price inflation.\n\n"
                f"**Verdict:** Walk away from this vendor or present our fair market estimation of **${predicted_fmv:,.2f}** as a non-negotiable counter-offer."
            )

        else:
            # 🟡 HOLD / NEGOTIATE LOGIC EXECUTIVE TEXT
            st.write(
                f"Our intelligent appraisal handler has assessed this vehicle. "
                f"The target asset is currently priced very accurately, floating inside a slim, normal market variance bracket of **{deviation*100:+.1f}%**. "
                f"This indicates the vehicle represents a fair transaction matching traditional operational timelines, with low room for extreme profit or loss margins."
            )
            
            st.markdown("#### ⚖ Balanced Valuation Tradeoffs")
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.write("**Value Strengths:**")
                for feat, val in pos_drivers:
                    st.write(f"✔ **{HUMAN_FEATURE_NAMES.get(feat, feat)}** increases appraisal baseline (+${val:,.2f}).")
            with col_neg:
                st.write("**Depreciation Weights:**")
                for feat, val in neg_drivers:
                    st.write(f"❌ **{HUMAN_FEATURE_NAMES.get(feat, feat)}** applies market markdown (-${abs(val):,.2f}).")

            st.markdown(" ")
            st.warning(
                f"💡 **Final Action Plan: HOLD & NEGOTIATE**\n\n"
                f"• The asking price matches industry reality closely.\n\n"
                f"• Use the depreciation adjustments found on its **{top_neg_name}** as safe leverage to negotiate minor repair coverage or price trims before closing escrow."
            )
    else:
        st.info("💡 Select target values on the left and click **Evaluate Deal** to view your automated, decision-specific analysis report.")