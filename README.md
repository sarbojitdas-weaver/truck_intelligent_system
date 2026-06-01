# 1. Overview
The Truck Price Intelligence System is a corporate-grade multi-model predictive dashboard designed to evaluate used truck valuations, project future residual assets, and provide clear investment indicators. It bridges rigorous historical training checkpoints (built across RandomForest, LightGBM, XGBoost, and CatBoost pipelines) directly to an interactive, real-time user workspace.

# 2. Key Features
- **Multi-Model Ensemble Valuations:** Aggregates individual subsystem architectures into generalized ensemble price points, neutralizing standalone algorithmic skew or bias.
- **Dynamic 3-Month Matrix Simulator:** Advances localized time features forward by 0.25 years and increases mileage factors safely to accurately forecast target vehicle depreciation curves.
- **Statistical Outlier-Bounded Scoring Engine:** Implements Interquartile Range (IQR) fence clipping rules to cleanly standardize multi-model margin alpha metrics into a normalized 0–100 investment rating.
- **Ensemble Spread Confidence Index:** Quantifies agreement stability across models by evaluating their underlying standard deviation and coefficient of variation relative to the calculated mean.
- **Real-Time Currency Layer:** Integrates dynamic sliders to safely normalize international asset data values (e.g., local tracking currency to standard USD displays).

# 3. Prerequisites & Requirements
The application layer relies on an isolated environment running Python 3.10+ along with specialized packages to handle processing, model tracking, and dashboard rendering:

```bash
# Required Dependencies
scikit-learn
lightgbm
xgboost
catboost
pandas
numpy
joblib
streamlit