import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

# Title
st.title("🎯 Employee Attrition Prediction")
st.markdown("Predict if an employee is likely to leave the company using Machine Learning")

# Load the trained pipeline (preferred) or fall back to previous model+encoders
@st.cache_resource
def load_model():
    # use repo-relative paths so deployment environments can find the files
    base = Path(__file__).resolve().parent
    pipeline_path = base / 'attrition_pipeline.pkl'
    model_path = base / 'logistic_model.pkl'
    encoders_path = base / 'label_encoders.pkl'

    if pipeline_path.exists():
        pipeline = pickle.load(open(pipeline_path, 'rb'))
        return pipeline, None

    # fallback (legacy)
    if model_path.exists() and encoders_path.exists():
        model = pickle.load(open(model_path, 'rb'))
        encoders = pickle.load(open(encoders_path, 'rb'))
        return model, encoders

    return None, None

model, label_encoders = load_model()

if model is None:
    st.error("No trained model found. Run the notebook cell that creates `attrition_pipeline.pkl` or add `logistic_model.pkl` and `label_encoders.pkl` to the project folder.")
    st.stop()

# Sidebar for input
st.sidebar.header("Employee Information")

# Numeric inputs
age = st.sidebar.slider("Age", 18, 65, 35)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
total_working_years = st.sidebar.slider("Total Working Years", 0, 50, 10)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 18, 2)
years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 17, 2)
distance_from_home = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
hourly_rate = st.sidebar.slider("Hourly Rate", 30, 100, 65)
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, 800)
monthly_rate = st.sidebar.slider("Monthly Rate", 2000, 27000, 14000)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike", 11, 25, 15)
training_times_last_year = st.sidebar.slider("Training Times Last Year", 0, 6, 2)
performance_rating = st.sidebar.slider("Performance Rating", 3, 4, 3)
education = st.sidebar.slider("Education Level", 1, 5, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 2)
job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 2)
job_level = st.sidebar.slider("Job Level", 1, 5, 2)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 2)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction", 1, 4, 2)
work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 2)

# Categorical inputs
st.sidebar.markdown("---")
department = st.sidebar.selectbox("Department", ["Sales", "HR", "Research & Development"])
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
"Manufacturing Director", "Healthcare Representative", "Manager", "Accountant", "Sales Representative", "Human Resources"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
over_time = st.sidebar.selectbox("Over Time", ["Yes", "No"])
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

# Extra (less-common) fields required by the pipeline — provide defaults or allow user input
education_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"], index=0)
employee_count = st.sidebar.number_input("Employee Count", min_value=1, max_value=10, value=1)
employee_number = st.sidebar.number_input("Employee Number (id)", min_value=1, value=1)
over_18 = st.sidebar.selectbox("Over18", ["Y"], index=0)
standard_hours = st.sidebar.number_input("Standard Hours", min_value=1, value=80)
stock_option_level = st.sidebar.number_input("Stock Option Level", min_value=0, max_value=3, value=0)

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'YearsAtCompany': [years_at_company],
    'TotalWorkingYears': [total_working_years],
    'YearsInCurrentRole': [years_in_current_role],
    'YearsSinceLastPromotion': [years_since_last_promotion],
    'YearsWithCurrManager': [years_with_curr_manager],
    'DistanceFromHome': [distance_from_home],
    'HourlyRate': [hourly_rate],
    'DailyRate': [daily_rate],
    'MonthlyRate': [monthly_rate],
    'NumCompaniesWorked': [num_companies_worked],
    'PercentSalaryHike': [percent_salary_hike],
    'TrainingTimesLastYear': [training_times_last_year],
    'PerformanceRating': [performance_rating],
    'Education': [education],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'JobInvolvement': [job_involvement],
    'JobLevel': [job_level],
    'JobSatisfaction': [job_satisfaction],
    'RelationshipSatisfaction': [relationship_satisfaction],
    'WorkLifeBalance': [work_life_balance],
    'Department': [department],
    'JobRole': [job_role],
    'Gender': [gender],
    'MaritalStatus': [marital_status],
    'OverTime': [over_time],
    'BusinessTravel': [business_travel],
    # additional columns expected by the pipeline
    'EducationField': [education_field],
    'EmployeeCount': [employee_count],
    'EmployeeNumber': [employee_number],
    'Over18': [over_18],
    'StandardHours': [standard_hours],
    'StockOptionLevel': [stock_option_level]
})

# Make prediction
if st.sidebar.button("🔮 Predict Attrition"):
    # If label_encoders is None we have a full pipeline
    if label_encoders is None:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    else:
        # legacy: apply saved encoders then predict with raw model
        for col in ['Department', 'JobRole', 'Gender', 'MaritalStatus', 'OverTime', 'BusinessTravel']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK OF ATTRITION")
            st.metric("Prediction", "Employee Likely to Leave", delta="Risk", delta_color="inverse")
        else:
            st.success("✅ LOW RISK OF ATTRITION")
            st.metric("Prediction", "Employee Likely to Stay", delta="Safe", delta_color="normal")
    
    with col2:
        st.metric("Probability of Attrition", f"{probability[1]*100:.2f}%")
        st.metric("Probability of Retention", f"{probability[0]*100:.2f}%")
    
    # Display input summary
    st.markdown("---")
    st.subheader("📊 Input Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Monthly Income:** ${monthly_income}")
        st.write(f"**Department:** {department}")
    with col2:
        st.write(f"**Years at Company:** {years_at_company}")
        st.write(f"**Job Role:** {job_role}")
        st.write(f"**Performance Rating:** {performance_rating}")
    with col3:
        st.write(f"**Job Satisfaction:** {job_satisfaction}/4")
        st.write(f"**Work Life Balance:** {work_life_balance}/4")
        st.write(f"**Over Time:** {over_time}")
