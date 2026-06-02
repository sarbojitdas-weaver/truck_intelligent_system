import streamlit as st
import numpy as np
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
# -------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Truck Price Intelligence System",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom layout adjustments for clean KPI card boundaries
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .buy-card { border-left-color: #28a745; background-color: #f1fbf4; }
    .avoid-card { border-left-color: #dc3545; background-color: #fdf2f4; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# RESOURCE CACHING: LOAD EMBEDDED PIPELINE ARTIFACTS
# -------------------------------------------------------------
@st.cache_resource
def load_pipeline_artifacts():
    try:
        model_path = hf_hub_download(
            repo_id="sarbojit-weavers/truck_price_prediction",
            filename="truck_price_prediction_model.pkl"
        )


        return joblib.load(model_path)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

artifacts = load_pipeline_artifacts()

if artifacts is not None:
    # Safely unpack model memory variables
    best_models = artifacts['best_models']
    trained_features = artifacts['trained_features']
    lower_fence = artifacts['lower_fence']
    upper_fence = artifacts['upper_fence']
    min_clipped_p = artifacts['min_clipped_p']
    max_clipped_p = artifacts['max_clipped_p']
    best_month_by_brand_year = artifacts['best_month_by_brand_year']
    row_template = artifacts['row_template']

    # -------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------
    st.title(" Truck Price Intelligence System")
    st.divider()

    # -------------------------------------------------------------
    # SIDEBAR CONTROL DECK (INPUT LAYER)
    # -------------------------------------------------------------
    st.sidebar.header(" Vehicle Parameter Matrix")

    input_brand = st.sidebar.selectbox(
        "Vehicle Brand",
        sorted(best_month_by_brand_year['brand'].unique())
    )



    
    # Standard widget structure to maintain continuous stream tracking
    input_year = st.sidebar.slider("Model Year", min_value=2000, max_value=2026, value=2018, step=1)
    input_mileage = st.sidebar.number_input("Mileage / Kilometers Driven (in Thousands)", min_value=1, max_value=2000, value=400, step=10, help="Specify value in thousands. E.g., 400 means 400,000.")
    input_price = st.sidebar.number_input("Listed Asking Purchase Price ($ USD)", min_value=500, max_value=500000, value=30000, step=1000)
    
    exchange_rate = st.sidebar.slider("Currency Exchange Rate (1 USD to INR)", min_value=70.0, max_value=90.0, value=83.0, step=0.5, help="Conversion metric to normalize local asset tracking valuations.")

    # Primary execution button
    run_eval = st.sidebar.button("Run Intelligence Engine Evaluations", type="primary")

    # -------------------------------------------------------------
    # PROCESS CALCULATIONS ONLY WHEN TRIGGERED
    # -------------------------------------------------------------
    if run_eval:
        with st.spinner("Synchronizing predictive timelines across machine learning systems..."):
            # 1. Adapt template row structure
            truck_row = row_template.iloc[[0]].copy()
            
            # 2. Map input variants seamlessly across case structures
            for col in truck_row.columns:
                if col.lower() == 'year': 
                    truck_row[col] = input_year
                elif col.lower() in ['mileage', 'kilometers_driven', 'km_driven', 'mileage_cleaned']:
                    truck_row[col] = input_mileage * 1000 if truck_row[col].max() > 5000 else input_mileage
                elif col.lower() == 'age': 
                    truck_row[col] = max(0, 2026 - input_year)
            
            # 3. Create Time-Shifted Twin Row (3-Month Target Frame)
            future_row = truck_row.copy()
            for col in future_row.columns:
                if col.lower() == 'age': 
                    future_row[col] += 0.25

            # 4. Generate Core Pipeline Model Output Matrix
            current_preds = [np.expm1(model.predict(truck_row[trained_features]))[0] for model in best_models.values()]
            future_preds = [np.expm1(model.predict(future_row[trained_features]))[0] for model in best_models.values()]
            
            # 5. Process Multi-Model Averages (Original Native INR Tracking)
            market_price_inr = np.mean(current_preds)
            future_price_inr = np.mean(future_preds)
            profit_inr = future_price_inr - market_price_inr
            
            # 6. Currency Normalization Layer (Local INR -> Display USD)
            market_price_usd = market_price_inr / exchange_rate
            future_price_usd = future_price_inr / exchange_rate
            profit_usd = profit_inr / exchange_rate

            # 7. Task 7 Boundary Rules: Normalizing Score (Using global IQR baseline bounds)
            clipped_p = np.clip(profit_inr, lower_fence, upper_fence)
            score = int(round(((clipped_p - min_clipped_p) / (max_clipped_p - min_clipped_p)) * 100))
            
            # 8. Task 8 Boundary Rules: Confidence Score via Model Coefficient of Variation Spread
            relative_spread = np.std(current_preds) / market_price_inr if market_price_inr > 0 else 0
            confidence = round(max(0.0, min(100.0, (1.0 - relative_spread) * 100)), 1)
            
            # 9. Task 10 Boundary Rules: Decision Matrix Mapping
            # 9. Recommendation Engine
            if score >= 70 and confidence >= 90:
                recommendation = "BUY"
                recommendation_icon = "🟢"
            else:
                recommendation = "AVOID"
                recommendation_icon = "🔴"

        match = best_month_by_brand_year[
            (best_month_by_brand_year['brand'] == input_brand) &
            (best_month_by_brand_year['year'] == input_year)
        ]

        if not match.empty:
            best_sell_time = match.iloc[0]['selling_window']

        else:
            brand_match = best_month_by_brand_year[
                best_month_by_brand_year['brand'] == input_brand
            ]

            if not brand_match.empty:
                best_sell_time = brand_match['selling_window'].mode()[0]
            else:
                best_sell_time = "N/A"

        # -------------------------------------------------------------
        # UI DISPLAY: OUTPUT LAYER & RESULTS MATRIX
        # -------------------------------------------------------------
        st.header("📊 Executive Investment Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.metric(
                        "Recommendation",
                        f"{recommendation_icon} {recommendation}"
                    )

        with col2:
                st.metric(
                    "Model Confidence",
                    f"{confidence}%"
                )

        with col3:
                st.metric(
                    "Opportunity Score",
                    f"{score}/100"
                )

        st.divider()

        st.subheader("💰 Market Valuation")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.metric(
                    "Current Market Value",
                    f"${market_price_usd:,.0f}"
                )

        with col2:
                st.metric(
                    "3-Month Forecast",
                    f"${future_price_usd:,.0f}"
                )

        with col3:
                st.metric(
                    "Expected Gain",
                    f"${profit_usd:,.2f}",
                    delta=f"${profit_usd:,.2f}"
                )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
                    st.metric(
                        "Recommended Selling Window",
                        best_sell_time
                    )
        with col2:
                st.metric(
                    "Forecast Trend",
                    "📈 Appreciating" if profit_usd > 0 else "📉 Stable / Depreciating"
                )

        st.divider()
        
       
  