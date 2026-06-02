import streamlit as st
import numpy as np
import pandas as pd
import joblib

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
        return joblib.load('truck_model_artifacts.pkl')
    except FileNotFoundError:
        st.error("'truck_model_artifacts.pkl' not found. Please export your notebook pipeline artifacts first.")
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
    best_overall_month = artifacts['best_overall_month']
    row_template = artifacts['row_template']

    # -------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------
    st.title(" Truck Price Intelligence System")
    st.subheader("Multi-Model Machine Learning Valuation & Ensemble Scoring Engine")
    st.divider()

    # -------------------------------------------------------------
    # SIDEBAR CONTROL DECK (INPUT LAYER)
    # -------------------------------------------------------------
    st.sidebar.header(" Vehicle Parameter Matrix")
    
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
            recommendation = "BUY" if score >= 50 and confidence >= 90.0 else "AVOID"

        # -------------------------------------------------------------
        # UI DISPLAY: OUTPUT LAYER & RESULTS MATRIX
        # -------------------------------------------------------------
        st.header("📊 Asset Investment Intelligence Report")
        
        # Row Layout 1: Primary Action Recommendation Alert Callouts
        col_rec, col_score, col_conf = st.columns(3)
        
        with col_rec:
            if recommendation == "BUY":
                st.markdown(f'<div class="metric-card buy-card"><h3>Strategic Recommendation</h3><h2>🟢 {recommendation}</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card avoid-card"><h3>Strategic Recommendation</h3><h2>🔴 {recommendation}</h2></div>', unsafe_allow_html=True)
                
        with col_score:
            st.markdown(f'<div class="metric-card"><h3>Investment Rating Score</h3><h2>{score} / 100</h2></div>', unsafe_allow_html=True)
            
        with col_conf:
            st.markdown(f'<div class="metric-card"><h3>Multi-Model Agreement Confidence</h3><h2>{confidence}%</h2></div>', unsafe_allow_html=True)

        st.subheader("💰 Fleet Valuation Estimates (Converted to USD)")
        
        # Row Layout 2: Capital Valuations Metrics Cards
        col_mkt, col_fut, col_prof, col_seas = st.columns(4)
        col_mkt.metric("Current Fair Market Price", f"${market_price_usd:,.2f}")
        col_fut.metric("Projected 3-Month Price", f"${future_price_usd:,.2f}")
        col_prof.metric("Expected Gross Margin / Growth", f"${profit_usd:,.2f}", delta=f"${profit_usd:,.2f}")
        col_seas.metric("Optimal Historical Exit Window", str(best_overall_month))

        st.divider()
        
        # Diagnostic Insights Container Expanders
        with st.expander("🔍 Model Verification Dashboard & System Diagnostics"):
            st.write("Review variance across isolated pipeline architectures prior to consolidation:")
            
            individual_summary_data = []
            for name, current, future in zip(best_models.keys(), current_preds, future_preds):
                ind_profit_usd = (future - current) / exchange_rate
                individual_summary_data.append({
                    "Subsystem Module Layout": name,
                    "Current Price Valuation ($)": f"${(current / exchange_rate):,.2f}",
                    "Future Price Projection ($)": f"${(future / exchange_rate):,.2f}",
                    "Calculated Alpha Delta ($)": f"${ind_profit_usd:,.2f}"
                })
            st.table(pd.DataFrame(individual_summary_data))
    else:
        st.info("💡 Adjust metrics on the left Sidebar control deck and click 'Run Intelligence Engine Evaluations' to generate your investment diagnostic report.")