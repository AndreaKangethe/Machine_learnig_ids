#!/usr/bin/env python3
import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your inference module
try:
    from mlids.src.inference import create_ids_inference
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False
    st.error("Neural network models not available. Please ensure inference.py is properly set up.")

def preprocess_input(df):
    """Basic preprocessing for input data."""
    # Handle non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

def format_prediction(pred, label_map):
    """Format prediction for display."""
    if isinstance(pred, (list, np.ndarray)):
        pred = pred[0]
    
    if pred in label_map:
        result = label_map[pred]
    else:
        result = "Attack" if pred == 1 else "Normal"
    
    return f"✅ {result}" if result == "Normal" else f"🚨 {result}"

# ===========================
# Load Neural Network Models
# ===========================
@st.cache_resource
def load_neural_models():
    """Load the trained neural network models."""
    if not NEURAL_MODELS_AVAILABLE:
        return None, "Neural network models not available"
    
    try:
        models_dir = os.path.join(project_root, "mlids", "src", "saved_models")
        ids_inference = create_ids_inference(models_dir)
        return ids_inference, None
    except Exception as e:
        return None, f"Error loading neural models: {str(e)}"

# ===========================
# Load Legacy Random Forest Model (Fallback)
# ===========================
@st.cache_data
def load_legacy_model():
    try:
        model_dir = os.path.join(current_dir, "saved_models")
        model_path = os.path.join(model_dir, "rf_balanced_model.pkl")

        if not os.path.exists(model_path):
            return None, None, None, None, "Legacy model file not found."

        saved = joblib.load(model_path)

        if isinstance(saved, dict) and "model" in saved:
            model = saved["model"]
            feature_names = saved.get("features", None)
            label_map = saved.get("label_map", {0: "Normal", 1: "Attack"})
        else:
            model = saved
            feature_names = None
            label_map = {0: "Normal", 1: "Attack"}

        # Load metrics
        metrics_path = os.path.join(model_dir, "performance_metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        return model, feature_names, label_map, metrics, None

    except Exception as e:
        return None, None, None, None, f"Error loading legacy model: {str(e)}"

# ===========================
# Load training data for feature ranges
# ===========================
@st.cache_data
def load_training_data():
    train_file = os.path.join(project_root, "mlids", "data", "processed", "kdd_train_balanced.csv")
    if os.path.exists(train_file):
        df = pd.read_csv(train_file)
        # Remove non-feature columns
        if 'binary_label' in df.columns:
            df = df.drop('binary_label', axis=1)
        if 'dataset' in df.columns:
            df = df.drop('dataset', axis=1)
        return df
    return None

def get_proper_feature_names():
    """Get the correct 44 feature names expected by the models."""
    return [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'feature_42', 'feature_43'  # Placeholder names for the remaining features
    ]

# ===========================
# Main Streamlit App
# ===========================
def main():
    st.set_page_config(page_title="ML-IDS Dashboard", layout="wide")
    st.title("🚨 ML-Based Intrusion Detection System")

    # Load models
    neural_ids, neural_error = load_neural_models()
    legacy_model, legacy_features, label_map, legacy_metrics, legacy_error = load_legacy_model()
    train_df = load_training_data()

    # Determine which models are available
    models_available = []
    if neural_ids is not None:
        models_available.append("Neural Network (FNN)")
        models_available.append("Autoencoder") 
        models_available.append("Ensemble (Both)")
    if legacy_model is not None:
        models_available.append("Random Forest (Legacy)")

    if not models_available:
        st.error("No models are available. Please check your model files.")
        st.write("Neural Network Error:", neural_error)
        st.write("Legacy Model Error:", legacy_error)
        return

    # Model selection
    selected_model = st.selectbox("Select Model", models_available, index=0)

    # Use proper feature names
    feature_names = get_proper_feature_names()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Real-Time Prediction", 
        "📂 Batch Prediction", 
        "📊 Model Performance", 
        "📈 Feature Analysis"
    ])

    # -------------------
    # Real-Time Prediction
    # -------------------
    with tab1:
        st.header("🔍 Real-Time Prediction")
        st.write(f"**Selected Model:** {selected_model}")
        
        live_update = st.checkbox("🔄 Enable Live Update / Auto-Predict", value=False)

        input_data = {}
        
        with st.expander("🛠️ Set Feature Values", expanded=True):
            # Create organized columns for better UX
            cols = st.columns(4)
            
            for i, feature_name in enumerate(feature_names):
                with cols[i % 4]:
                    if train_df is not None and len(train_df.columns) > i:
                        # Use training data statistics for realistic ranges
                        col_data = train_df.iloc[:, i]
                        if pd.api.types.is_numeric_dtype(col_data):
                            min_val = float(col_data.min())
                            max_val = float(col_data.max())
                            median_val = float(col_data.median())
                            
                            # Prevent min_val == max_val issues
                            if min_val >= max_val:
                                input_data[feature_name] = st.number_input(
                                    label=f"{feature_name} ({i})",
                                    value=median_val,
                                    key=f"feature_{i}"
                                )
                            else:
                                step_val = max((max_val - min_val) / 100, 0.01)
                                input_data[feature_name] = st.slider(
                                    label=f"{feature_name} ({i})",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=median_val,
                                    step=step_val,
                                    key=f"feature_{i}",
                                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                                )
                        else:
                            # Handle categorical features
                            unique_vals = col_data.unique()
                            input_data[feature_name] = st.selectbox(
                                label=f"{feature_name} ({i})",
                                options=unique_vals,
                                key=f"feature_{i}"
                            )
                    else:
                        # Default input for missing training data
                        input_data[feature_name] = st.slider(
                            label=f"{feature_name} ({i})",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                            key=f"feature_{i}"
                        )

        def predict_sample():
            try:
                # Create DataFrame with proper feature names
                sample_df = pd.DataFrame([input_data])
                
                # Make prediction based on selected model
                if "Neural Network" in selected_model and neural_ids:
                    result = neural_ids.predict_single(sample_df, model_type='fnn')
                    prediction_text = result['prediction_label']
                    confidence = result['confidence']
                    risk_level = result['risk_level']
                    
                    color = "green" if prediction_text == "Normal" else "red"
                    st.markdown(f"<h2 style='color:{color};'>🎯 {prediction_text}</h2>", unsafe_allow_html=True)
                    st.write(f"**Confidence:** {confidence:.2%}")
                    st.write(f"**Risk Level:** {risk_level}")
                    
                elif "Autoencoder" in selected_model and neural_ids:
                    result = neural_ids.predict_single(sample_df, model_type='autoencoder')
                    prediction_text = result['prediction_label']
                    confidence = result['confidence']
                    
                    color = "green" if prediction_text == "Normal" else "red"
                    st.markdown(f"<h2 style='color:{color};'>🔍 {prediction_text}</h2>", unsafe_allow_html=True)
                    st.write(f"**Anomaly Score:** {confidence:.4f}")
                    
                elif "Ensemble" in selected_model and neural_ids:
                    result = neural_ids.predict_single(sample_df, model_type='ensemble')
                    prediction_text = result['prediction_label']
                    confidence = result['confidence']
                    risk_level = result['risk_level']
                    
                    color = "green" if prediction_text == "Normal" else "red"
                    st.markdown(f"<h2 style='color:{color};'>🎯 {prediction_text} (Ensemble)</h2>", unsafe_allow_html=True)
                    st.write(f"**Confidence:** {confidence:.2%}")
                    st.write(f"**Risk Level:** {risk_level}")
                    
                elif "Random Forest" in selected_model and legacy_model:
                    # Use legacy model
                    processed = preprocess_input(sample_df.copy())
                    
                    # Ensure we have the right number of features
                    if len(processed.columns) > len(feature_names):
                        processed = processed.iloc[:, :len(feature_names)]
                    elif len(processed.columns) < len(feature_names):
                        # Pad with zeros
                        for i in range(len(processed.columns), len(feature_names)):
                            processed[f'feature_{i}'] = 0
                    
                    pred = legacy_model.predict(processed)
                    result_label = format_prediction(pred, label_map)
                    
                    color = "green" if "Normal" in result_label else "red"
                    st.markdown(f"<h2 style='color:{color};'>🌳 {result_label}</h2>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Debug info:")
                st.write(f"Sample shape: {sample_df.shape if 'sample_df' in locals() else 'Not created'}")
                st.write(f"Selected model: {selected_model}")
                st.write(f"Neural models available: {neural_ids is not None}")

        # Prediction trigger
        if live_update:
            predict_sample()
        else:
            if st.button("🔍 Predict", type="primary"):
                predict_sample()

    # -------------------
    # Batch Prediction
    # -------------------
    with tab2:
        st.header("📂 Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("### Uploaded Data Preview")
                st.dataframe(data.head())

                if st.button("🔍 Run Batch Prediction"):
                    with st.spinner("Processing..."):
                        if "Neural" in selected_model or "Ensemble" in selected_model:
                            # Use neural network models
                            method = 'ensemble' if 'Ensemble' in selected_model else 'fnn'
                            results = neural_ids.predict_ensemble(data, method=method)
                            
                            data["Prediction"] = ['Attack' if p == 1 else 'Normal' for p in results['ensemble']['predictions']]
                            data["Confidence"] = results['ensemble']['confidence_scores']
                            
                            st.write(f"**Results Summary:**")
                            st.write(f"- Normal Traffic: {results['ensemble']['normal_count']}")
                            st.write(f"- Attacks Detected: {results['ensemble']['attack_count']}")
                            
                        else:
                            # Use legacy model
                            processed = preprocess_input(data.copy())
                            predictions = legacy_model.predict(processed)
                            data["Prediction"] = [format_prediction([p], label_map) for p in predictions]

                        st.write("### Results Preview")
                        st.dataframe(data[['Prediction'] + (['Confidence'] if 'Confidence' in data.columns else [])].head())

                        # Visualization
                        pred_counts = data['Prediction'].value_counts()
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                                   title="Prediction Distribution")
                        st.plotly_chart(fig, use_container_width=True)

                        # Download button
                        csv = data.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Batch prediction error: {str(e)}")

    # -------------------
    # Model Performance
    # -------------------
    with tab3:
        st.header("📊 Model Performance Metrics")
        
        if "Neural" in selected_model or "Ensemble" in selected_model:
            st.subheader("🧠 Neural Network Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**FNN Model:**")
                st.write("- Accuracy: 99.90%")
                st.write("- Precision: 100%")
                st.write("- Recall: 99.79%")
                st.write("- F1-Score: 99.90%")
                st.write("- ROC-AUC: 100%")
                
            with col2:
                st.write("**Autoencoder Model:**")
                st.write("- Accuracy: 97.70%")
                st.write("- ROC-AUC: 99.74%")
                st.write("- Detection Rate: 98.54%")
                st.write("- False Alarm Rate: 3.14%")
                
        elif legacy_metrics:
            st.subheader("🌳 Random Forest Performance")
            st.json(legacy_metrics)
        else:
            st.warning("No performance metrics available for the selected model.")

    # -------------------
    # Feature Analysis
    # -------------------
    with tab4:
        st.header("📈 Feature Analysis")
        
        if "Random Forest" in selected_model and legacy_model:
            try:
                if hasattr(legacy_model, 'feature_importances_'):
                    importances = legacy_model.feature_importances_
                    fi_df = pd.DataFrame({
                        "Feature": feature_names[:len(importances)],
                        "Importance": importances
                    }).sort_values("Importance", ascending=False)
                    
                    st.dataframe(fi_df.head(20))
                    
                    fig = px.bar(fi_df.head(20), x="Importance", y="Feature", 
                               orientation="h", title="Top 20 Feature Importances")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Selected model doesn't support feature importance analysis.")
            except Exception as e:
                st.error(f"Feature analysis error: {str(e)}")
        else:
            st.write("Feature importance analysis is currently available for Random Forest model only.")
            if train_df is not None:
                st.subheader("📊 Feature Statistics from Training Data")
                st.dataframe(train_df.describe())

if __name__ == "__main__":
    main()
