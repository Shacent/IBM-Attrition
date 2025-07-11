# streamlit_app_with_login.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Prediction App - Secure",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# User credentials (in production, use database)
USER_CREDENTIALS = {
    "admin": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # password: admin123
    "user": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",   # password: user123
    "demo": "2a97516c354b68848cdbd8f54a226a0a55b21ed138e207ad6c5cbb9c00aa5aea"    # password: demo123
}

USER_ROLES = {
    "admin": "Administrator",
    "user": "Standard User", 
    "demo": "Demo User"
}

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h1>🔐 ML Prediction App</h1>
        <h3>Secure Machine Learning Platform</h3>
        <p>Please login to access the prediction system</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🚪 Login")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submit_button:
                if username in USER_CREDENTIALS:
                    if verify_password(password, USER_CREDENTIALS[username]):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = USER_ROLES[username]
                        st.success(f"✅ Welcome {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid password!")
                else:
                    st.error("❌ Username not found!")
        
        # Demo credentials info
        with st.expander("📋 Demo Credentials", expanded=False):
            st.markdown("""
            **Test Accounts:**
            
            🔹 **Admin Account:**
            - Username: `admin`
            - Password: `admin123`
            
            🔹 **User Account:**
            - Username: `user` 
            - Password: `user123`
            
            🔹 **Demo Account:**
            - Username: `demo`
            - Password: `demo123`
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <small>🛡️ Secure ML Platform v1.0<br>
            Powered by Streamlit & Machine Learning</small>
        </div>
        """, unsafe_allow_html=True)

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.rerun()

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

def create_input_form(feature_names, user_role):
    """Create input form for user to enter feature values"""
    st.sidebar.header(f"📊 Input Features")
    st.sidebar.markdown(f"**User:** {st.session_state.username} ({user_role})")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="secondary"):
        logout()
    
    st.sidebar.markdown("---")
    
    # Create input fields for each feature
    input_data = {}
    
    # Role-based feature access
    if user_role == "Demo User":
        # Demo users get limited features
        limited_features = feature_names[:10]  # First 10 features only
        st.sidebar.info("🔒 Demo account: Limited to 10 features")
        features_to_use = limited_features
    else:
        features_to_use = feature_names
    
    for i, feature in enumerate(features_to_use):
        if feature.lower() in ['age', 'year', 'time', 'duration']:
            input_data[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=0,
                max_value=100,
                value=25,
                step=1,
                key=f"input_{i}"
            )
        elif feature.lower() in ['price', 'amount', 'cost', 'income', 'salary']:
            input_data[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
                key=f"input_{i}"
            )
        elif feature.lower() in ['rate', 'percentage', 'ratio', 'score']:
            input_data[feature] = st.sidebar.slider(
                f"{feature.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key=f"input_{i}"
            )
        else:
            input_data[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                value=0.0,
                step=0.1,
                key=f"input_{i}"
            )
    
    # Fill missing features with 0 for demo users
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0.0
    
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

def display_prediction_results(prediction, prediction_proba, user_role):
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
        
        # Role-based additional info
        if user_role == "Administrator":
            st.subheader("🔧 Admin Info")
            st.write(f"Raw probabilities: {prediction_proba}")
            st.write(f"User role: {user_role}")
    
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

def display_model_info(metadata, user_role):
    """Display model information and performance metrics"""
    st.subheader("🤖 Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", metadata.get('model_type', 'N/A'))
        st.metric("Training Date", str(metadata.get('training_date', 'N/A'))[:10])
    
    with col2:
        test_acc = metadata.get('test_accuracy', 'N/A')
        if isinstance(test_acc, (int, float)):
            st.metric("Test Accuracy", f"{test_acc:.4f}")
        else:
            st.metric("Test Accuracy", str(test_acc))
            
        roc_auc = metadata.get('roc_auc', 'N/A')
        if isinstance(roc_auc, (int, float)):
            st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        else:
            st.metric("ROC-AUC Score", str(roc_auc))
    
    with col3:
        cv_score = metadata.get('cv_score', 'N/A')
        if isinstance(cv_score, (int, float)):
            st.metric("CV Score", f"{cv_score:.4f}")
        else:
            st.metric("CV Score", str(cv_score))
            
        train_time = metadata.get('training_time', 'N/A')
        if isinstance(train_time, (int, float)):
            st.metric("Training Time", f"{train_time:.2f}s")
        else:
            st.metric("Training Time", str(train_time))
    
    with col4:
        st.metric("Features Used", metadata.get('feature_count', 'N/A'))
        st.metric("User Role", user_role)

def main_app():
    """Main application after login"""
    user_role = st.session_state.user_role
    username = st.session_state.username
    
    # App title with user info
    st.title(f"🤖 ML Prediction App - Welcome {username}!")
    st.markdown(f"**Role:** {user_role} | **Session:** Active 🟢")
    st.markdown("---")
    
    # Load model and components
    model, scaler, feature_names, metadata = load_model_and_scaler()
    
    # Display model information
    display_model_info(metadata, user_role)
    st.markdown("---")
    
    # Create input form
    st.header("📝 Input Your Data")
    input_data = create_input_form(feature_names, user_role)
    
    # Main prediction section
    st.header("🔮 Make Prediction")
    
    if st.button("🚀 Predict", type="primary"):
        with st.spinner("Making prediction..."):
            prediction, prediction_proba = make_prediction(model, scaler, input_data, feature_names)
            
            if prediction is not None:
                st.success("Prediction completed successfully!")
                display_prediction_results(prediction, prediction_proba, user_role)
            else:
                st.error("Failed to make prediction. Please check your input data.")
    
    # Feature importance section (Admin only)
    if user_role == "Administrator":
        st.markdown("---")
        st.header("📊 Feature Importance (Admin Only)")
        
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
    st.write(f"""
    1. **Input Features**: Use the sidebar to input your feature values
    2. **Make Prediction**: Click the 'Predict' button to get results  
    3. **View Results**: See the prediction result and confidence level
    4. **Role Access**: Your role ({user_role}) determines available features
    """)

def main():
    """Main function to control app flow"""
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # Route to appropriate page
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()
