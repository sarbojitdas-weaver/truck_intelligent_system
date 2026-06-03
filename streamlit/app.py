
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Truck Price Intelligence System",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------------------
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
.buy-card {
    border-left-color: #28a745;
    background-color: #f1fbf4;
}
.avoid-card {
    border-left-color: #dc3545;
    background-color: #fdf2f4;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# -------------------------------------------------------------
@st.cache_resource
def load_pipeline_artifacts():

    try:
        
        model_path = hf_hub_download(
            repo_id="sarbojit-weavers/price_prediction",
            filename="price_prediction_model.pkl"
        )
        # model_path="price_prediction_model.pkl"

        artifacts = joblib.load(model_path)

        return artifacts

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


artifacts = load_pipeline_artifacts()

# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
if artifacts is not None:

    # -------------------------------------------------------------
    # UNPACK ARTIFACTS
    # -------------------------------------------------------------
    best_models = artifacts['best_models']
    trained_features = artifacts['trained_features']

    lower_fence = artifacts['lower_fence']
    upper_fence = artifacts['upper_fence']

    min_clipped_p = artifacts['min_clipped_p']
    max_clipped_p = artifacts['max_clipped_p']

    best_month_by_brand_year = artifacts['best_month_by_brand_year']

    row_template = artifacts['row_template']

    # -------------------------------------------------------------
    # TITLE
    # -------------------------------------------------------------
    st.title("🚛 Truck Price Intelligence System")
    st.divider()

    # -------------------------------------------------------------
    # SIDEBAR INPUTS
    # -------------------------------------------------------------
    st.sidebar.header("Vehicle Parameter Matrix")

    input_brand = st.sidebar.selectbox(
        "Vehicle Brand",
        sorted(best_month_by_brand_year['brand'].unique())
    )

    input_year = st.sidebar.slider(
        "Model Year",
        min_value=2000,
        max_value=2026,
        value=2018,
        step=1
    )

    input_mileage = st.sidebar.number_input(
        "Mileage / Kilometers Driven (Thousands)",
        min_value=1,
        max_value=2000,
        value=400,
        step=10,
        help="400 means 400,000 KM"
    )

    input_price = st.sidebar.number_input(
        "Listed Asking Purchase Price ($ USD)",
        min_value=500,
        max_value=500000,
        value=30000,
        step=1000
    )

    exchange_rate = st.sidebar.slider(
        "USD → INR Exchange Rate",
        min_value=70.0,
        max_value=90.0,
        value=83.0,
        step=0.5
    )

    run_eval = st.sidebar.button(
        "Run Intelligence Engine Evaluations",
        type="primary"
    )

    # -------------------------------------------------------------
    # EXECUTE PREDICTIONS
    # -------------------------------------------------------------
    if run_eval:

        with st.spinner("Synchronizing predictive timelines..."):

            # -------------------------------------------------------------
            # BRAND-SPECIFIC TEMPLATE ROW
            # -------------------------------------------------------------
            if 'brand' in row_template.columns:

                brand_match = row_template[
                    row_template['brand'].str.lower() ==
                    input_brand.lower()
                ]

            elif 'Brand' in row_template.columns:

                brand_match = row_template[
                    row_template['Brand'].str.lower() ==
                    input_brand.lower()
                ]

            else:
                brand_match = pd.DataFrame()

            if not brand_match.empty:
                truck_row = brand_match.iloc[[0]].copy()
            else:
                truck_row = row_template.iloc[[0]].copy()

            # -------------------------------------------------------------
            # FEATURE INJECTION
            # -------------------------------------------------------------
            for col in truck_row.columns:

                col_lower = col.lower()

                # ---------------------------------------------------------
                # YEAR
                # ---------------------------------------------------------
                if col_lower == 'year':
                    truck_row[col] = input_year

                # ---------------------------------------------------------
                # BRAND
                # ---------------------------------------------------------
                elif col_lower == 'brand':
                    truck_row[col] = input_brand

                # ---------------------------------------------------------
                # PRICE
                # ---------------------------------------------------------
                elif col_lower in [
                    'price',
                    'askprice',
                    'listing_price',
                    'price_cleaned',
                    'sellingprice'
                ]:
                    truck_row[col] = input_price * exchange_rate

                # ---------------------------------------------------------
                # MILEAGE
                # ---------------------------------------------------------
                elif col_lower in [
                    'mileage',
                    'kilometers_driven',
                    'km_driven',
                    'mileage_cleaned',
                    'odometer'
                ]:
                    truck_row[col] = input_mileage * 1000

                # ---------------------------------------------------------
                # AGE
                # ---------------------------------------------------------
                elif col_lower == 'age':
                    truck_row[col] = max(0, 2026 - input_year)

            # -------------------------------------------------------------
            # DERIVED FEATURES
            # -------------------------------------------------------------
            actual_price_inr = input_price * exchange_rate
            actual_mileage = input_mileage * 1000

            if 'price_per_mile' in truck_row.columns:

                truck_row['price_per_mile'] = (
                    actual_price_inr / (actual_mileage + 1)
                )

            # -------------------------------------------------------------
            # FUTURE ROW CREATION
            # -------------------------------------------------------------
            future_row = truck_row.copy()

            month_map = {
                'jan': 1,
                'feb': 2,
                'mar': 3,
                'apr': 4,
                'may': 5,
                'jun': 6,
                'jul': 7,
                'aug': 8,
                'sep': 9,
                'oct': 10,
                'nov': 11,
                'dec': 12
            }

            reverse_month = {
                1: 'Jan',
                2: 'Feb',
                3: 'Mar',
                4: 'Apr',
                5: 'May',
                6: 'Jun',
                7: 'Jul',
                8: 'Aug',
                9: 'Sep',
                10: 'Oct',
                11: 'Nov',
                12: 'Dec'
            }

            # -------------------------------------------------------------
            # MONTH SHIFT
            # -------------------------------------------------------------
            if 'month' in truck_row.columns:

                current_month = str(
                    truck_row['month'].values[0]
                ).lower()[:3]

                current_month_num = month_map.get(current_month, 1)

                future_month_num = (
                    (current_month_num + 3 - 1) % 12
                ) + 1

                future_month_str = reverse_month[future_month_num]

            else:
                future_month_num = 1
                future_month_str = 'Jan'

            # -------------------------------------------------------------
            # FUTURE SEASON
            # -------------------------------------------------------------
            if future_month_num in [12, 1, 2]:
                future_season = 'Winter'

            elif future_month_num in [3, 4, 5]:
                future_season = 'Spring'

            elif future_month_num in [6, 7, 8]:
                future_season = 'Summer'

            else:
                future_season = 'Autumn'

            # -------------------------------------------------------------
            # APPLY FUTURE SHIFTS
            # -------------------------------------------------------------
            for col in future_row.columns:

                col_lower = col.lower()

                if col_lower == 'age':
                    future_row[col] += 0.25

                elif col_lower == 'month':
                    future_row[col] = future_month_str

                elif col_lower == 'season':
                    future_row[col] = future_season

            # -------------------------------------------------------------
            # MODEL PREDICTIONS
            # -------------------------------------------------------------
            current_preds = [
                np.expm1(
                    model.predict(truck_row[trained_features])
                )[0]
                for model in best_models.values()
            ]

            future_preds = [
                np.expm1(
                    model.predict(future_row[trained_features])
                )[0]
                for model in best_models.values()
            ]

            # -------------------------------------------------------------
            # AGGREGATED VALUES
            # -------------------------------------------------------------
            market_price_inr = np.mean(current_preds)

            future_price_inr = np.mean(future_preds)

            profit_inr = (
                future_price_inr - market_price_inr
            )

            # -------------------------------------------------------------
            # USD CONVERSION
            # -------------------------------------------------------------
            market_price_usd = (
                market_price_inr / exchange_rate
            )

            future_price_usd = (
                future_price_inr / exchange_rate
            )

            profit_usd = (
                profit_inr / exchange_rate
            )

            # -------------------------------------------------------------
            # OPPORTUNITY SCORE
            # -------------------------------------------------------------
            clipped_p = np.clip(
                profit_inr,
                lower_fence,
                upper_fence
            )

            score = int(round(
                (
                    (clipped_p - min_clipped_p)
                    /
                    (max_clipped_p - min_clipped_p)
                ) * 100
            ))

            # -------------------------------------------------------------
            # CONFIDENCE SCORE
            # -------------------------------------------------------------
            relative_spread = (
                np.std(current_preds)
                / market_price_inr
                if market_price_inr > 0
                else 0
            )

            confidence = round(
                max(
                    0.0,
                    min(
                        100.0,
                        (1.0 - relative_spread) * 100
                    )
                ),
                1
            )

            # -------------------------------------------------------------
            # RECOMMENDATION ENGINE
            # -------------------------------------------------------------
            if profit_usd >0 and profit_inr>0:

                recommendation = "BUY"
                recommendation_icon = "🟢"

            else:

                recommendation = "AVOID"
                recommendation_icon = "🔴"

            # -------------------------------------------------------------
            # BEST SELLING WINDOW
            # -------------------------------------------------------------
            match = best_month_by_brand_year[
                (
                    best_month_by_brand_year['brand']
                    == input_brand
                )
                &
                (
                    best_month_by_brand_year['year']
                    == input_year
                )
            ]

            if not match.empty:

                best_sell_time = (
                    match.iloc[0]['selling_window']
                )

            else:

                brand_match = best_month_by_brand_year[
                    best_month_by_brand_year['brand']
                    == input_brand
                ]

                if not brand_match.empty:

                    best_sell_time = (
                        brand_match['selling_window']
                        .mode()[0]
                    )

                else:

                    best_sell_time = "N/A"

        # -------------------------------------------------------------
        # OUTPUT SECTION
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

        # -------------------------------------------------------------
        # VALUATION
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # SELLING WINDOW
        # -------------------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Recommended Selling Window",
                best_sell_time
            )

        with col2:
            st.metric(
                "Forecast Trend",
                (
                    "📈 Appreciating"
                    if profit_usd > 0
                    else "📉 Depreciating"
                )
            )

        st.divider()

        # -------------------------------------------------------------
        # DEBUG PANEL
        # -------------------------------------------------------------
        with st.expander("🔍 Debug Feature Snapshot"):

            st.write("Current Row")
            st.dataframe(truck_row)

            st.write("Future Row")
            st.dataframe(future_row)