"""
Streamlit Web App for Immo Eliza ML
Simple, user-friendly interface for property price prediction.
"""

import streamlit as st
import pandas as pd
from immo_eliza_ml.predict import Predict
import os

# Page config
st.set_page_config(
    page_title="Immo Eliza ML - Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Immo Eliza ML - Belgian Real Estate Price Predictor")
st.markdown("Predict property prices using machine learning models trained on Belgian real estate data.")

# Initialize predictor (cache it to avoid reloading on every interaction)
@st.cache_resource
def load_predictor():
    """Load the predictor once and cache it."""
    try:
        predictor = Predict(
            models_folder="models",
            preprocessor_path="models/preprocessor.json"
        )
        predictor.load()
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure you've trained the models first by running: `python -m immo_eliza_ml.main`")
        return None

predictor = load_predictor()

if predictor is None:
    st.stop()

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
model_options = list(predictor.models.keys())
selected_model = st.sidebar.selectbox(
    "Select Model",
    model_options,
    index=model_options.index("XGBoost") if "XGBoost" in model_options else 0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"**Selected:** {selected_model}\n\n**Available Models:** {len(model_options)}")

# Main input form
st.header("📝 Property Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    postal_code = st.number_input("Postal Code", min_value=1000, max_value=9999, value=9000, step=1)
    living_area = st.number_input("Living Area (m²)", min_value=1.0, value=120.0, step=1.0)
    number_of_rooms = st.number_input("Number of Rooms", min_value=1, value=3, step=1)
    number_of_facades = st.number_input("Number of Facades", min_value=1, value=2, step=1)

with col2:
    st.subheader("Features & Amenities")
    equipped_kitchen = st.selectbox("Equipped Kitchen", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
    furnished = st.selectbox("Furnished", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")
    open_fire = st.selectbox("Open Fire", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")
    terrace = st.selectbox("Terrace", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")
    garden = st.selectbox("Garden", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
    swimming_pool = st.selectbox("Swimming Pool", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Outdoor Spaces")
    garden_surface = st.number_input("Garden Surface (m²)", min_value=0.0, value=150.0, step=1.0)
    terrace_surface = st.number_input("Terrace Surface (m²)", min_value=0.0, value=0.0, step=1.0)

with col4:
    st.subheader("Property Details")
    state_of_building = st.selectbox(
        "State of Building",
        ["good", "as_new", "to_renovate", "to_be_done_up", "just_renovated"],
        index=0
    )
    subtype_of_property = st.selectbox(
        "Subtype of Property",
        ["house", "apartment", "villa", "mansion", "duplex", "studio"],
        index=0
    )

# Prediction button
st.markdown("---")
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    # Prepare property data
    property_data = {
        "postal_code": int(postal_code),
        "living_area": float(living_area),
        "number_of_rooms": int(number_of_rooms),
        "number_of_facades": int(number_of_facades),
        "equipped_kitchen": int(equipped_kitchen),
        "furnished": int(furnished),
        "open_fire": int(open_fire),
        "terrace": int(terrace),
        "garden": int(garden),
        "swimming_pool": int(swimming_pool),
        "garden_surface": float(garden_surface),
        "terrace_surface": float(terrace_surface),
        "state_of_building": state_of_building,
        "subtype_of_property": subtype_of_property,
    }
    
    try:
        # Single prediction
        price = predictor.predict_single(property_data, model_name=selected_model)
        
        # Display result
        st.success(f"### 💰 Predicted Price: **€{price:,.0f}**")
        st.info(f"Using model: **{selected_model}**")
        
        # Compare all models
        st.markdown("---")
        st.subheader("📊 Comparison Across All Models")
        
        all_predictions = predictor.predict_all_models(property_data)
        
        comparison_df = pd.DataFrame([
            {"Model": name, "Predicted Price (€)": f"{price:,.0f}"}
            for name, price in all_predictions.items()
        ])
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Statistics
        prices = list(all_predictions.values())
        stats = predictor.predict_with_confidence(property_data)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Mean", f"€{stats['mean']:,.0f}")
        with col_stat2:
            st.metric("Median", f"€{stats['median']:,.0f}")
        with col_stat3:
            st.metric("Range", f"€{stats['min']:,.0f} - €{stats['max']:,.0f}")
        with col_stat4:
            st.metric("Std Dev", f"€{stats['std']:,.0f}")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Immo Eliza ML - Machine Learning Price Prediction</p>
        <p>Built with Streamlit | Models trained on Belgian real estate data</p>
    </div>
    """,
    unsafe_allow_html=True
)

