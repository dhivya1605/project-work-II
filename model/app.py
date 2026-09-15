"""
model/app.py
------------
Streamlit Web Application & Research Dashboard for Crop Recommendation and SHAP Explainability.
"""

import sys
import os

MODEL_DIR = os.path.abspath(os.path.dirname(__file__))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from predict import CropPredictor, FEATURE_DISPLAY_NAMES
from shap_explainer import CropSHAPExplainer

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation & XAI Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    /* Global styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #86EFAC;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .crop-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #15803D;
        margin: 0.2rem 0;
        text-transform: capitalize;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #166534;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .section-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_pipeline():
    """
    Cached initializer for CropPredictor and CropSHAPExplainer.
    Ensures model and artifacts are loaded only once.
    """
    try:
        artifacts_path = os.path.join(MODEL_DIR, "artifacts")
        predictor = CropPredictor(artifacts_dir=artifacts_path)
        explainer = CropSHAPExplainer(predictor)
        return predictor, explainer, None
    except Exception as e:
        return None, None, str(e)


def main():
    st.markdown('<div class="main-header">🌾 Explainable Crop Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deep tabular inference powered by <b>TabKANet</b>, optimized via <b>SFOA</b> & <b>DLO</b> with <b>SHAP</b> local explainability.</div>',
        unsafe_allow_html=True,
    )

    predictor, explainer, load_err = load_ml_pipeline()

    if load_err:
        st.error(f"❌ Failed to load trained model artifacts: {load_err}")
        st.info("Please verify that `model/artifacts/` contains `tabkanet_weights.pt`, `scaler.pkl`, `label_encoder.pkl`, and `model_config.json`.")
        st.stop()

    # Preset agronomic samples
    PRESETS = {
        "Custom Input": None,
        "Rice Preset (High Water, High Temp)": {"N": 90, "P": 42, "K": 43, "SOIL_PH": 6.5, "TEMP": 27.5, "WATERREQUIRED": 1850, "RELATIVE_HUMIDITY": 82, "CROPDURATION": 130},
        "Wheat Preset (Moderate Temp, Dry)": {"N": 60, "P": 35, "K": 35, "SOIL_PH": 6.8, "TEMP": 18.0, "WATERREQUIRED": 450, "RELATIVE_HUMIDITY": 55, "CROPDURATION": 110},
        "Groundnut Preset (Acidic Soil, Warm)": {"N": 25, "P": 50, "K": 30, "SOIL_PH": 5.8, "TEMP": 28.0, "WATERREQUIRED": 500, "RELATIVE_HUMIDITY": 60, "CROPDURATION": 120},
        "Cotton Preset (Alkaline, Warm)": {"N": 120, "P": 50, "K": 50, "SOIL_PH": 7.8, "TEMP": 32.0, "WATERREQUIRED": 700, "RELATIVE_HUMIDITY": 60, "CROPDURATION": 160},
        "Ragi Preset (Dry, Neutral Soil)": {"N": 40, "P": 20, "K": 20, "SOIL_PH": 6.5, "TEMP": 25.0, "WATERREQUIRED": 350, "RELATIVE_HUMIDITY": 50, "CROPDURATION": 100},
        "Sugarcane Preset (High Nutrient, Humid)": {"N": 150, "P": 60, "K": 90, "SOIL_PH": 6.8, "TEMP": 30.0, "WATERREQUIRED": 1500, "RELATIVE_HUMIDITY": 80, "CROPDURATION": 300},
    }

    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Agronomic Parameters")
        
        preset_choice = st.selectbox("Quick Load Preset Sample", list(PRESETS.keys()))
        selected_preset = PRESETS[preset_choice]

        st.subheader("1. Soil Nutrients (N, P, K)")
        n_val = st.number_input("Nitrogen (N) [kg/ha]", min_value=0.0, max_value=300.0, value=float(selected_preset["N"]) if selected_preset else 80.0, step=1.0)
        p_val = st.number_input("Phosphorus (P) [kg/ha]", min_value=0.0, max_value=200.0, value=float(selected_preset["P"]) if selected_preset else 45.0, step=1.0)
        k_val = st.number_input("Potassium (K) [kg/ha]", min_value=0.0, max_value=300.0, value=float(selected_preset["K"]) if selected_preset else 40.0, step=1.0)
        ph_val = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=float(selected_preset["SOIL_PH"]) if selected_preset else 6.5, step=0.1)

        st.subheader("2. Climate & Environment")
        temp_val = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=float(selected_preset["TEMP"]) if selected_preset else 26.0, step=0.5)
        water_val = st.number_input("Rainfall / Water Req. (mm)", min_value=0.0, max_value=4000.0, value=float(selected_preset["WATERREQUIRED"]) if selected_preset else 1100.0, step=10.0)

        with st.expander("Additional Environment Parameters", expanded=False):
            rh_val = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=float(selected_preset["RELATIVE_HUMIDITY"]) if selected_preset else 75.0, step=1.0)
            duration_val = st.number_input("Crop Duration (Days)", min_value=10.0, max_value=365.0, value=float(selected_preset["CROPDURATION"]) if selected_preset else 120.0, step=5.0)

        st.markdown("---")
        predict_btn = st.button("🚀 Predict Crop & Generate SHAP", type="primary", use_container_width=True)

    input_dict = {
        "N": n_val,
        "P": p_val,
        "K": k_val,
        "SOIL_PH": ph_val,
        "TEMP": temp_val,
        "WATERREQUIRED": water_val,
        "RELATIVE_HUMIDITY": rh_val,
        "CROPDURATION": duration_val,
    }

    # Input validation guardrails
    validation_errors = []
    if ph_val < 3.5 or ph_val > 9.5:
        validation_errors.append(f"Soil pH value ({ph_val}) is outside typical agricultural limits (3.5 – 9.5).")
    if n_val < 0 or p_val < 0 or k_val < 0:
        validation_errors.append("Soil nutrient values (N, P, K) cannot be negative.")
    if water_val < 0:
        validation_errors.append("Rainfall / Water required cannot be negative.")

    if validation_errors:
        for err in validation_errors:
            st.warning(f"⚠️ {err}")

    # Tabs layout
    tab1, tab2, tab3 = st.tabs(["📊 Recommendation & SHAP Explanation", "🏗️ Model Architecture & Benchmark", "📄 Full SHAP Data Table"])

    with tab1:
        if predict_btn or "last_prediction" not in st.session_state:
            with st.spinner("Running TabKANet model inference & computing SHAP feature contributions..."):
                try:
                    pred_res = predictor.predict(input_dict)
                    shap_res = explainer.explain(input_dict, target_class_idx=pred_res["predicted_class_index"])
                    st.session_state["last_prediction"] = pred_res
                    st.session_state["last_shap"] = shap_res
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

        pred_res = st.session_state.get("last_prediction")
        shap_res = st.session_state.get("last_shap")

        if pred_res and shap_res:
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:1.1rem; color:#4B5563; font-weight:600;">RECOMMENDED CROP</div>
                        <div class="crop-title">🌱 {pred_res['predicted_crop'].upper()}</div>
                        <div style="margin-top:0.8rem;">
                            <span class="confidence-badge">Confidence: {pred_res['confidence_pct']}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_res2:
                st.subheader("Top 5 Crop Probabilities")
                top_crops = pred_res["top_k_crops"]
                df_top = pd.DataFrame(top_crops, columns=["Crop", "Probability"])
                df_top["Probability_Pct"] = df_top["Probability"] * 100

                fig_top = px.bar(
                    df_top,
                    x="Probability_Pct",
                    y="Crop",
                    orientation="h",
                    text=df_top["Probability_Pct"].apply(lambda v: f"{v:.1f}%"),
                    color="Probability_Pct",
                    color_continuous_scale="Greens",
                )
                fig_top.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Confidence Probability (%)",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_top, use_container_width=True)

            st.markdown("---")
            st.header("🔍 Why was this crop recommended?")
            st.caption("SHAP (SHapley Additive exPlanations) attribution for the predicted class:")

            shap_df = shap_res["shap_table"].copy()

            # Plot SHAP Horizontal Waterfall/Bar Chart
            shap_df_sorted = shap_df.sort_values("SHAP_Value", ascending=True)

            colors = ["#22C55E" if v > 0 else "#EF4444" for v in shap_df_sorted["SHAP_Value"]]

            fig_shap = go.Figure()
            fig_shap.add_trace(
                go.Bar(
                    y=shap_df_sorted["Feature_Name"],
                    x=shap_df_sorted["SHAP_Value"],
                    orientation="h",
                    marker=dict(color=colors),
                    text=[f"{v:+.4f} (Input: {u})" for v, u in zip(shap_df_sorted["SHAP_Value"], shap_df_sorted["User_Value"])],
                    textposition="auto",
                )
            )

            fig_shap.update_layout(
                title=f"SHAP Feature Attribution towards recommending <b>{pred_res['predicted_crop'].upper()}</b>",
                xaxis_title="SHAP Value (Positive = Favors Crop, Negative = Pushes Away)",
                yaxis_title="Feature",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Natural language summary callout
            st.info(f"💡 **Explainability Insight:** {shap_res['explanation_text']}")

    with tab2:
        st.header("⚙️ Research Project Architecture")
        
        cfg = predictor.config
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Test Accuracy", f"{cfg.get('test_accuracy', 0.9646)*100:.2f}%")
        with col_m2:
            st.metric("Validation Accuracy", f"{cfg.get('final_val_accuracy', 0.9826)*100:.2f}%")
        with col_m3:
            st.metric("Total Crop Classes", cfg.get("num_classes", 49))

        st.markdown("---")
        st.subheader("1. SFOA Selected Features")
        st.write("Superb Fairy-wren Optimization Algorithm selected the optimal numeric feature subset:")
        st.code(cfg.get("selected_columns", []), language="json")

        st.subheader("2. DLO Tuned Hyperparameters")
        st.write("Draco Lizard Optimizer discovered the best TabKANet hyperparameters:")
        st.json(cfg.get("hyperparameters", {}))

        st.subheader("3. Pipeline Workflow")
        st.code(
            """
[ Input Features: N, P, K, pH, Rainfall, Temp, RH, Duration ]
               │
               ▼
[ SFOA Feature Subset Extraction: TEMP, WATERREQUIRED, RH, Duration ]
               │
               ▼
[ StandardScaler Normalization (scaler.pkl) ]
               │
               ▼
[ FastKAN Numerical Embedding + Transformer Backbone (TabKANet) ]
               │
               ▼
[ Softmax Classification (49 Crop Classes) ]
               │
               ▼
[ Model-Agnostic KernelSHAP Attribution Explainer ]
            """,
            language="text",
        )

    with tab3:
        st.header("📄 Detailed SHAP Values Table")
        if shap_res:
            display_table = shap_res["shap_table"][["Feature_Name", "Feature_Code", "User_Value", "SHAP_Value", "Impact"]].copy()
            st.dataframe(
                display_table.style.highlight_max(axis=0, subset=["SHAP_Value"], color="#DCFCE7")
                .highlight_min(axis=0, subset=["SHAP_Value"], color="#FEE2E2"),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
