# HR Employee Attrition Prediction - Streamlit App

## Overview
This is a Streamlit web application that predicts employee attrition using a Logistic Regression machine learning model trained on HR employee data.

## Project Structure
```
├── ibm.ipynb                  # Jupyter notebook with model training code
├── app.py                     # Streamlit application
├── logistic_model.pkl         # Saved trained logistic regression model
├── label_encoders.pkl         # Saved label encoders for categorical variables
├── requirements.txt           # Python dependencies
└── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
```

## Installation

1. **Clone or download the project folder**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

4. The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Input Employee Information** in the left sidebar:
   - Numeric values (age, income, years, etc.)
   - Categorical selections (department, job role, gender, etc.)

2. **Click "Predict Attrition"** to get the prediction

3. **View Results:**
   - Risk status (High Risk or Low Risk)
   - Probability percentages
   - Input summary

## Model Details

- **Algorithm:** Logistic Regression
- **Training Data:** 1470 HR employees
- **Features:** 34 (after encoding categorical variables)
- **Target Variable:** Attrition (Yes/No)

## Features Used

**Numeric Features:**
- Age, MonthlyIncome, YearsAtCompany, TotalWorkingYears, etc.

**Categorical Features:**
- Department, JobRole, Gender, MaritalStatus, OverTime, BusinessTravel

## Deployment Options

### Deploy on Streamlit Cloud (Free)
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo and deploy

### Deploy Locally
```bash
streamlit run app.py
```

### Deploy on Heroku/AWS/Azure
1. Follow their respective documentation
2. Include requirements.txt for dependencies
3. Ensure model and encoder pickle files are in the deployment directory

## Notes
- The app uses cached model loading for better performance
- Label encoders are pre-loaded and applied to categorical inputs
- All predictions are made in real-time based on user inputs

## Future Improvements
- Add feature importance visualization
- Include model accuracy metrics
- Add batch prediction from CSV
- Implement data validation
- Add more model comparison options

