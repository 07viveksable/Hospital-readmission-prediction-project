"""
Hospital Readmission Prediction Web App
Streamlit Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff8e1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        return model, scaler, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, model_info = load_model()

# Header
st.title("🏥 Hospital Readmission Risk Prediction System")
st.markdown("### Predict 30-day readmission risk for diabetic patients")
st.markdown("---")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    
    if model_info:
        st.metric("Model Type", model_info.get('model_name', 'N/A'))
        st.metric("ROC-AUC Score", f"{model_info['performance']['roc_auc']:.4f}")
        st.metric("Accuracy", f"{model_info['performance']['accuracy']:.4f}")
        
        st.markdown("---")
        st.markdown("**Training Date:**")
        st.write(model_info.get('training_date', 'N/A'))
    
    st.markdown("---")
    st.markdown("### 🎯 Risk Levels")
    st.markdown("🟢 **Low Risk** (<30%)")
    st.markdown("🟡 **Medium Risk** (30-60%)")
    st.markdown("🔴 **High Risk** (>60%)")

# Main content
if model is None:
    st.error("⚠️ Model not loaded. Please ensure model files are in 'models/' directory.")
else:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Analytics"])
    
    # ============================================================
    # TAB 1: SINGLE PREDICTION
    # ============================================================
    with tab1:
        st.header("Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 0, 100, 65, help="Patient age in years")
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            st.subheader("Hospital Stay")
            time_in_hospital = st.slider("Length of Stay (days)", 1, 14, 5)
            
        with col2:
            st.subheader("Medical Procedures")
            num_lab_procedures = st.number_input("Lab Procedures", 0, 150, 50)
            num_procedures = st.number_input("Medical Procedures", 0, 10, 3)
            number_diagnoses = st.number_input("Number of Diagnoses", 1, 16, 9)
            
        with col3:
            st.subheader("Medications & History")
            num_medications = st.number_input("Number of Medications", 0, 81, 15)
            medication_changed = st.checkbox("Medication Changed During Visit")
            diabetic_med_prescribed = st.checkbox("Diabetic Medication Prescribed")
            
            st.subheader("Prior Visits")
            number_outpatient = st.number_input("Outpatient Visits (past year)", 0, 42, 0)
            number_emergency = st.number_input("Emergency Visits (past year)", 0, 76, 0)
            number_inpatient = st.number_input("Inpatient Visits (past year)", 0, 21, 0)
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 PREDICT READMISSION RISK", use_container_width=True):
            
            # Prepare input data
            gender_encoded = 1 if gender == "Male" else 0
            total_visits = number_outpatient + number_emergency + number_inpatient
            had_emergency = 1 if number_emergency > 0 else 0
            
            input_data = pd.DataFrame({
                'time_in_hospital': [time_in_hospital],
                'num_lab_procedures': [num_lab_procedures],
                'num_procedures': [num_procedures],
                'num_medications': [num_medications],
                'number_outpatient': [number_outpatient],
                'number_emergency': [number_emergency],
                'number_inpatient': [number_inpatient],
                'number_diagnoses': [number_diagnoses],
                'age_numeric': [age],
                'gender_encoded': [gender_encoded],
                'medication_changed': [1 if medication_changed else 0],
                'diabetic_med_prescribed': [1 if diabetic_med_prescribed else 0],
                'total_visits': [total_visits],
                'had_emergency': [had_emergency]
            })
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
                risk_color = "🟢"
                risk_class = "risk-low"
            elif probability < 0.6:
                risk_level = "Medium"
                risk_color = "🟡"
                risk_class = "risk-medium"
            else:
                risk_level = "High"
                risk_color = "🔴"
                risk_class = "risk-high"
            
            # Display results
            st.markdown("---")
            st.markdown("## 📋 Prediction Results")
            
            # Risk card
            st.markdown(f"""
                <div class="{risk_class}">
                    <h2>{risk_color} {risk_level} Risk</h2>
                    <h3>Readmission Probability: {probability:.1%}</h3>
                    <p style="font-size: 18px;">
                        {'This patient is <b>likely to be readmitted</b> within 30 days.' if prediction == 1 else 'This patient is <b>unlikely to be readmitted</b> within 30 days.'}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visual gauge
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Risk Gauge")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Readmission Risk %", 'font': {'size': 24}},
                    delta = {'reference': 50, 'increasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#c8e6c9'},
                            {'range': [30, 60], 'color': '#fff9c4'},
                            {'range': [60, 100], 'color': '#ffcdd2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Key Factors")
                st.metric("Hospital Stay", f"{time_in_hospital} days")
                st.metric("Medications", num_medications)
                st.metric("Diagnoses", number_diagnoses)
                st.metric("Prior Visits", total_visits)
            
            # Clinical recommendations
            st.markdown("---")
            st.subheader("🏥 Clinical Recommendations")
            
            if risk_level == "High":
                st.error(f"""
                **HIGH RISK ({probability:.1%}) - IMMEDIATE ACTION REQUIRED**
                
                ✅ Schedule follow-up within **3-7 days**
                
                ✅ Assign dedicated care coordinator
                
                ✅ Arrange home health care services
                
                ✅ Provide detailed discharge instructions with teach-back method
                
                ✅ Ensure medication reconciliation completed
                
                ✅ Confirm patient has support system at home
                
                ✅ Consider telehealth monitoring
                """)
                
            elif risk_level == "Medium":
                st.warning(f"""
                **MEDIUM RISK ({probability:.1%}) - ENHANCED MONITORING**
                
                ✅ Schedule follow-up within **14 days**
                
                ✅ Provide comprehensive discharge education
                
                ✅ Verify patient understands all medications
                
                ✅ Ensure patient has primary care physician contact
                
                ✅ Provide written discharge summary
                """)
                
            else:
                st.success(f"""
                **LOW RISK ({probability:.1%}) - STANDARD PROTOCOL**
                
                ✅ Standard discharge protocol
                
                ✅ Routine follow-up as medically indicated
                
                ✅ Provide standard discharge materials
                
                ✅ Emergency contact information provided
                """)
    
    # ============================================================
    # TAB 2: BATCH PREDICTION
    # ============================================================
    with tab2:
        st.header("📊 Batch Prediction from CSV")
        
        st.markdown("""
        Upload a CSV file with patient data. Required columns:
        - time_in_hospital, num_lab_procedures, num_procedures, num_medications
        - number_outpatient, number_emergency, number_inpatient, number_diagnoses
        - age_numeric, gender_encoded, medication_changed, diabetic_med_prescribed
        - total_visits, had_emergency
        """)
        
        uploaded_file = st.file_uploader("diabetic_data.csv", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df_batch = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File loaded: {len(df_batch)} patients")
                
                st.subheader("Preview Data")
                st.dataframe(df_batch.head())
                
                if st.button("🔮 Predict for All Patients"):
                    
                    # Scale and predict
                    X_batch = scaler.transform(df_batch)
                    predictions = model.predict(X_batch)
                    probabilities = model.predict_proba(X_batch)[:, 1]
                    
                    # Add results to dataframe
                    df_batch['Prediction'] = ['Readmit' if p == 1 else 'No Readmit' for p in predictions]
                    df_batch['Probability'] = probabilities
                    df_batch['Risk_Level'] = pd.cut(
                        probabilities,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    st.markdown("---")
                    st.subheader("📈 Batch Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Patients", len(df_batch))
                    with col2:
                        st.metric("High Risk", sum(df_batch['Risk_Level'] == 'High'))
                    with col3:
                        st.metric("Medium Risk", sum(df_batch['Risk_Level'] == 'Medium'))
                    with col4:
                        st.metric("Low Risk", sum(df_batch['Risk_Level'] == 'Low'))
                    
                    # Risk distribution chart
                    fig = px.pie(
                        df_batch,
                        names='Risk_Level',
                        title='Risk Distribution',
                        color='Risk_Level',
                        color_discrete_map={
                            'Low': '#4caf50',
                            'Medium': '#ff9800',
                            'High': '#f44336'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results table
                    st.subheader("Detailed Results")
                    st.dataframe(df_batch)
                    
                    # Download button
                    csv = df_batch.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="readmission_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # ============================================================
    # TAB 3: ANALYTICS (FIXED INDENTATION)
    # ============================================================
    with tab3:
        st.header("📈 Model Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Performance")
            
            if model_info:
                performance = model_info['performance']
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Score': [
                        performance['accuracy'],
                        performance['precision'],
                        performance['recall'],
                        performance['f1_score'],
                        performance['roc_auc']
                    ]
                })
                
                fig = px.bar(
                    metrics_df,
                    x='Metric',
                    y='Score',
                    title='Model Performance Metrics',
                    color='Score',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Feature Importance")
            
            try:
                if hasattr(model, 'feature_importances_'):
                    # Feature names
                    feature_names = [
                        'Length of Stay', 'Lab Procedures', 'Medical Procedures',
                        'Number of Medications', 'Outpatient Visits', 'Emergency Visits',
                        'Inpatient Visits', 'Number of Diagnoses', 'Age',
                        'Gender', 'Medication Changed', 'Diabetic Med Prescribed',
                        'Total Visits', 'Had Emergency Visit'
                    ]
                    
                    # Get importances
                    importances = model.feature_importances_
                    
                    # Create dataframe
                    feature_imp = pd.DataFrame({
                        'Feature': feature_names[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Top 10
                    top_10 = feature_imp.head(10)
                    
                    # Bar chart
                    fig = go.Figure(go.Bar(
                        x=top_10['Importance'],
                        y=top_10['Feature'],
                        orientation='h',
                        marker=dict(
                            color=top_10['Importance'],
                            colorscale='Blues',
                            showscale=False
                        )
                    ))
                    
                    fig.update_layout(
                        title='Top 10 Most Important Features',
                        xaxis_title='Importance Score',
                        yaxis_title='',
                        height=400,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top 5 in text
                    st.markdown("**Most Important Factors:**")
                    for i, (idx, row) in enumerate(top_10.head(5).iterrows(), 1):
                        percentage = (row['Importance'] / importances.sum()) * 100
                        st.markdown(f"{i}. **{row['Feature']}**: {row['Importance']:.4f} ({percentage:.1f}%)")
                        
                else:
                    st.info("🔍 Feature importance is only available for tree-based models (Random Forest, XGBoost)")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not display feature importance: {str(e)}")
                st.info("Make sure your model is a tree-based model (Random Forest or XGBoost)")
        
        # Display images if available
        st.markdown("---")
        st.subheader("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                st.image('visualizations/confusion_matrix.png', caption='Confusion Matrix')
            except:
                st.info("Confusion matrix not available")
        
        with col2:
            try:
                st.image('visualizations/roc_curve.png', caption='ROC Curve')
            except:
                st.info("ROC curve not available")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Hospital Readmission Prediction System | Powered by Machine Learning</p>
        <p style='font-size: 12px;'>⚠️ For research and educational purposes only. Not intended for clinical use without validation.</p>
    </div>
""", unsafe_allow_html=True)