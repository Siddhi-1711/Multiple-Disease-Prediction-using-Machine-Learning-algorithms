from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
app = Flask(__name__)


# Load the models for diabetes
diabetes_svm_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/diabetes/svm_model.pkl'
diabetes_logistic_regression_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/diabetes/logistic_regression_model.pkl'
diabetes_random_forest_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/diabetes/random_forest_model.pkl'

with open(diabetes_svm_model_path, 'rb') as file:
    diabetes_svm_model = pickle.load(file)
with open(diabetes_logistic_regression_model_path, 'rb') as file:
    diabetes_logistic_regression_model = pickle.load(file)
with open(diabetes_random_forest_model_path, 'rb') as file:
    diabetes_random_forest_model = pickle.load(file)

# Load the models for cancer
cancer_svm_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/cancer/svm_model.pkl'
cancer_logistic_regression_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/cancer/logistic_regression_model.pkl'
cancer_random_forest_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/cancer/random_forest_model.pkl'

with open(cancer_svm_model_path, 'rb') as file:
    cancer_svm_model = pickle.load(file)
with open(cancer_logistic_regression_model_path, 'rb') as file:
    cancer_logistic_regression_model = pickle.load(file)
with open(cancer_random_forest_model_path, 'rb') as file:
    cancer_random_forest_model = pickle.load(file)

# Direct paths to the trained models
svm_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/heart_disease/svm_model.pkl'
logistic_regression_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/heart_disease/logistic_regression_model.pkl'
random_forest_model_path = 'C:/Users/SIDDHI/Desktop/Multiple-Disease-Prediction/app/models/heart_disease/random_forest_model.pkl'

# Load the models
with open(svm_model_path, 'rb') as file:
    svm_model = pickle.load(file)
with open(logistic_regression_model_path, 'rb') as file:
    logistic_regression_model = pickle.load(file)
with open(random_forest_model_path, 'rb') as file:
    random_forest_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/heart_disease', methods=['POST'])
def predict_heart_disease():
# Extract data from the form
    age = float(request.form['age']) # Convert to float
    sex = 1 if request.form['sex'] == 'Male' else 0 # Convert 'Male' to 1 and 'Female' to 0
    cp = 1 if request.form['cp'] == 'Non-Anginal Pain' else 0 # Convert 'Non-Anginal Pain' to 1, others to 0
    trestbps = float(request.form['trestbps']) # Convert to float
    chol = float(request.form['chol']) # Convert to float
    fbs = 1 if request.form['fbs'] == 'Greater than 120 mg/dl' else 0 # Convert 'Greater than 120 mg/dl' to 1, others to 0
    restecg = 1 if request.form['restecg'] == 'Normal' else 0 # Convert 'Normal' to 1, others to 0
    thalach = float(request.form['thalach']) # Convert to float
    exang = 1 if request.form['exang'] == 'Yes' else 0 # Convert 'Yes' to 1 and 'No' to 0
    oldpeak = float(request.form['oldpeak']) # Convert to float
    slope = 1 if request.form['slope'] == 'Flat' else 0 # Convert 'Flat' to 1, others to 0
    ca = float(request.form['ca']) # Convert to float
    thal = 1 if request.form['thal'] == 'Fixed Defect' else 0 # Convert 'Fixed Defect' to 1, others to 0

    # Prepare the data for prediction
    # Ensure the data is in a DataFrame with the correct column names
    prediction_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Make predictions with all models
    svm_prediction = svm_model.predict(prediction_data)
    logistic_regression_prediction = logistic_regression_model.predict(prediction_data)
    random_forest_prediction = random_forest_model.predict(prediction_data)
    # After extracting and preparing the data


    # Decide the final risk level based on the predictions
    high_risk_predictions = sum([svm_prediction[0] == 1, logistic_regression_prediction[0] == 1, random_forest_prediction[0] == 1])
    if high_risk_predictions >= 2:
        risk_level = "High Risk of Heart Disease"
    else:
        risk_level = "Low Risk of Heart Disease"

    recommendations = {
    "High Risk of Heart Disease": [
        "Consider reducing your intake of saturated fats.",
        "Increase your physical activity. Aim for at least 30 minutes of moderate-intensity exercise most days of the week.",
        "Consult with a healthcare professional for a personalized plan.",
        "Monitor your cholesterol levels regularly.",
        "Consider taking a statin if advised by your doctor."
    ],
    "Low Risk of Heart Disease": [
        "Maintain a healthy diet rich in fruits, vegetables, and whole grains.",
        "Engage in regular physical activity.",
        "Limit alcohol consumption.",
        "Regularly monitor your blood pressure and cholesterol levels.",
        "Consider consulting with a healthcare professional for preventive care."
    ]
    }
    # Get the recommendations based on the risk level
    recommendations_list = recommendations[risk_level]

    # Render the result template with the risk_level and recommendations
    return render_template('result.html', risk_level=risk_level, recommendations=recommendations_list)


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    # Extract data from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['bloodPressure'])
    skin_thickness = float(request.form['skinThickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
    age = float(request.form['age'])

    # Dictionary for feature transformation
    dic = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    # Transform features
    dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
            'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
            'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}

    if dic['BMI'] <= 18.5:
        dic2['NewBMI_Underweight'] = 1
    elif 18.5 < dic['BMI'] <= 24.9:
        pass
    elif 24.9 < dic['BMI'] <= 29.9:
        dic2['NewBMI_Overweight'] = 1
    elif 29.9 < dic['BMI'] <= 34.9:
        dic2['NewBMI_Obesity 1'] = 1
    elif 34.9 < dic['BMI'] <= 39.9:
        dic2['NewBMI_Obesity 2'] = 1
    elif dic['BMI'] > 39.9:
        dic2['NewBMI_Obesity 3'] = 1

    if 16 <= dic['Insulin'] <= 166:
        dic2['NewInsulinScore_Normal'] = 1

    if dic['Glucose'] <= 70:
        dic2['NewGlucose_Low'] = 1
    elif 70 < dic['Glucose'] <= 99:
        dic2['NewGlucose_Normal'] = 1
    elif 99 < dic['Glucose'] <= 126:
        dic2['NewGlucose_Overweight'] = 1
    elif dic['Glucose'] > 126:
        dic2['NewGlucose_Secret'] = 1

    dic.update(dic2)
    prediction_data = pd.DataFrame([dic.values()], columns=dic.keys())

    # Make predictions with all models
    svm_prediction = diabetes_svm_model.predict(prediction_data)
    logistic_regression_prediction = diabetes_logistic_regression_model.predict(prediction_data)
    random_forest_prediction = diabetes_random_forest_model.predict(prediction_data)

    
    # Decide the final risk level based on the predictions
    high_risk_predictions = sum([svm_prediction[0] == 1, logistic_regression_prediction[0] == 1, random_forest_prediction[0] == 1])


    if high_risk_predictions >= 2:
        risk_level = "High Risk of Diabetes"
    else:
        risk_level = "Low Risk of Diabetes"

    # Recommendations based on risk level
    recommendations = {
        "High Risk of Diabetes": [
            "Consult with a healthcare professional for a personalized plan.",
            "Monitor your blood sugar levels regularly.",
            "Consider taking medication if advised by your doctor."
        ],
        "Low Risk of Diabetes": [
            "Maintain a healthy diet rich in fruits, vegetables, and whole grains.",
            "Engage in regular physical activity.",
            "Limit alcohol consumption."
        ]
    }
    recommendations_list = recommendations[risk_level]
    
    # Render the result template with the risk_level and recommendations
    return render_template('result.html', risk_level=risk_level, recommendations=recommendations_list)


@app.route('/predict/cancer', methods=['POST'])
def predict_cancer():
    # Extract data from the form
    texture_mean = float(request.form['texture_mean'])
    smoothness_mean = float(request.form['smoothness_mean'])
    compactness_mean = float(request.form['compactness_mean'])
    concavity_mean = float(request.form['concave_points_mean'])
    symmetry_mean = float(request.form['symmetry_mean'])
    fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
    texture_se = float(request.form['texture_se'])
    area_se = float(request.form['area_se'])
    smoothness_se = float(request.form['smoothness_se'])
    compactness_se = float(request.form['compactness_se'])
    concavity_se = float(request.form['concavity_se'])
    concave_points_se = float(request.form['concave_points_se'])
    symmetry_se = float(request.form['symmetry_se'])
    fractal_dimension_se = float(request.form['fractal_dimension_se'])
    texture_worst = float(request.form['texture_worst'])
    area_worst = float(request.form['area_worst'])
    smoothness_worst = float(request.form['smoothness_worst'])
    compactness_worst = float(request.form['compactness_worst'])
    concavity_worst = float(request.form['concavity_worst'])
    concave_points_worst = float(request.form['concave_points_worst'])
    symmetry_worst = float(request.form['symmetry_worst'])
    fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

    # Prepare the data for prediction
    prediction_data = pd.DataFrame([[texture_mean, smoothness_mean, compactness_mean, concavity_mean, symmetry_mean, fractal_dimension_mean, texture_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, texture_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]], columns=['texture_mean', 'smoothness_mean', 'compactness_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'texture_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'])

    # Make predictions with the cancer model
    svm_prediction = cancer_svm_model.predict(prediction_data)
    logistic_regression_prediction = cancer_logistic_regression_model.predict(prediction_data)
    random_forest_prediction = cancer_random_forest_model.predict(prediction_data)

    # Decide the final risk level based on the predictions
    high_risk_predictions = sum([svm_prediction[0] == 1, logistic_regression_prediction[0] == 1, random_forest_prediction[0] == 1])
    if high_risk_predictions >= 2:
        risk_level = "High Risk of Cancer"
    else:
        risk_level = "Low Risk of Cancer"

    # Recommendations based on risk level
    recommendations = {
        "High Risk of Cancer": [
            "Consult with a healthcare professional for a personalized plan.",
            "Undergo regular screenings as advised by your doctor.",
            "Consider taking preventive medications if advised by your doctor."
        ],
        "Low Risk of Cancer": [
            "Maintain a healthy lifestyle.",
            "Engage in regular physical activity.",
            "Limit alcohol consumption."
        ]
    }
    recommendations_list = recommendations[risk_level]

    # Render the result template with the risk_level and recommendations
    return render_template('result.html', risk_level=risk_level, recommendations=recommendations_list)

if __name__ == '__main__':
    app.run(debug=True)
