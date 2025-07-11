# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model, scaler, and feature names"""
    try:
        # Load model
        model = joblib.load('models/logistic_regression_model.pkl')
        
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Load feature names
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, feature_names, metadata
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def create_input_form(feature_names):
    """Create input form for user to enter feature values"""
    st.sidebar.header("📊 Input Features")
    
    # Create input fields for each feature
    input_data = {}
    
    # You can customize these based on your actual features
    # For demonstration, I'll create generic inputs
    for feature in feature_names:
        if feature.lower() in ['age', 'year', 'time', 'duration']:
            # Numeric features that might be integers
            input_data[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=0,
                max_value=100,
                value=25,
                step=1
            )
        elif feature.lower() in ['price', 'amount', 'cost', 'income', 'salary']:
            # Numeric features that might be floats
            input_data[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0
            )
        elif feature.lower() in ['rate', 'percentage', 'ratio', 'score']:
            # Percentage or ratio features
            input_data[feature] = st.sidebar.slider(
                f"{feature.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01
            )
        else:
            # Default numeric input
            input_data[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                value=0.0,
                step=0.1
            )
    
    return input_data

def make_prediction(model, scaler, input_data, feature_names):
    """Make prediction based on input data"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def display_prediction_results(prediction, prediction_proba):
    """Display prediction results with visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.success(f"**Prediction: Positive Class** ✅")
        else:
            st.info(f"**Prediction: Negative Class** ❌")
        
        st.subheader("📊 Prediction Confidence")
        confidence = max(prediction_proba) * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with col2:
        st.subheader("📈 Probability Distribution")
        
        # Create probability bar chart
        prob_df = pd.DataFrame({
            'Class': ['Negative (0)', 'Positive (1)'],
            'Probability': prediction_proba
        })
        
        fig = px.bar(
            prob_df, 
            x='Class', 
            y='Probability',
            title='Prediction Probabilities',
            color='Probability',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_model_info(metadata):
    """Display model information and performance metrics"""
    st.subheader("🤖 Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", metadata['model_type'])
        st.metric("Training Date", metadata['training_date'][:10])
    
    with col2:
        st.metric("Test Accuracy", f"{metadata['test_accuracy']:.4f}")
        st.metric("ROC-AUC Score", f"{metadata['roc_auc']:.4f}")
    
    with col3:
        st.metric("CV Score", f"{metadata['cv_score']:.4f}")
        st.metric("Training Time", f"{metadata['training_time']:.2f}s")
    
    with col4:
        st.metric("Features Used", metadata['feature_count'])
        st.metric("Model Size", "Lightweight")

def main():
    """Main Streamlit application"""
    
    # App title and description
    st.title("🤖 Machine Learning Prediction App")
    st.markdown("---")
    st.write("This app uses a trained Logistic Regression model to make predictions based on your input features.")
    
    # Load model and components
    model, scaler, feature_names, metadata = load_model_and_scaler()
    
    # Display model information
    display_model_info(metadata)
    st.markdown("---")
    
    # Create input form
    st.header("📝 Input Your Data")
    input_data = create_input_form(feature_names)
    
    # Main prediction section
    st.header("🔮 Make Prediction")
    
    if st.button("🚀 Predict", type="primary"):
        with st.spinner("Making prediction..."):
            prediction, prediction_proba = make_prediction(model, scaler, input_data, feature_names)
            
            if prediction is not None:
                st.success("Prediction completed successfully!")
                display_prediction_results(prediction, prediction_proba)
            else:
                st.error("Failed to make prediction. Please check your input data.")
    
    # Feature importance section (if available)
    st.markdown("---")
    st.header("📊 Feature Importance")
    
    try:
        # Get feature coefficients for logistic regression
        feature_importance = abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(10), 
            x='Importance', 
            y='Feature',
            title='Top 10 Most Important Features',
            orientation='h'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning("Feature importance visualization not available.")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.write("""
    1. **Input Features**: Use the sidebar to input your feature values
    2. **Make Prediction**: Click the 'Predict' button to get results
    3. **View Results**: See the prediction result and confidence level
    4. **Interpret**: Use the probability distribution to understand the model's confidence
    """)
    
    st.markdown("### ⚠️ Important Notes")
    st.write("""
    - This model is trained on historical data and predictions are not guaranteed
    - Always validate results with domain expertise
    - For production use, consider retraining with fresh data
    """)

if __name__ == "__main__":
    main()