from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    with open('patel.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Encoding maps
binary_map = {'Yes': 1, 'No': 0}
gender_map = {'Female': 0, 'Male': 1}
multi_lines_map = {'No phone service': 0, 'No': 1, 'Yes': 2}
internet_services = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_map = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}
no_internet_map = {'No internet service': 0, 'No': 1, 'Yes': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Encode the input data
        input_data = {
            'gender': gender_map[data['gender']],
            'SeniorCitizen': int(data['SeniorCitizen']),
            'Partner': binary_map[data['Partner']],
            'Dependents': binary_map[data['Dependents']],
            'tenure': int(data['tenure']),
            'PhoneService': binary_map[data['PhoneService']],
            'MultipleLines': multi_lines_map[data['MultipleLines']],
            'InternetService': internet_services[data['InternetService']],
            'OnlineSecurity': no_internet_map[data['OnlineSecurity']],
            'OnlineBackup': no_internet_map[data['OnlineBackup']],
            'DeviceProtection': no_internet_map[data['DeviceProtection']],
            'TechSupport': no_internet_map[data['TechSupport']],
            'StreamingTV': no_internet_map[data['StreamingTV']],
            'StreamingMovies': no_internet_map[data['StreamingMovies']],
            'Contract': contract_map[data['Contract']],
            'PaperlessBilling': binary_map[data['PaperlessBilling']],
            'PaymentMethod': payment_map[data['PaymentMethod']],
            'MonthlyCharges': float(data['MonthlyCharges']),
            'TotalCharges': float(data['TotalCharges'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype(float)
        
        # Make prediction
        if model is not None:
            probability = model.predict_proba(input_df)[0][1]
            prediction = int(probability > 0.5)
            
            result = {
                'prediction': prediction,
                'probability': float(probability),
                'status': 'success'
            }
        else:
            result = {
                'status': 'error',
                'message': 'Model not loaded'
            }
            
    except Exception as e:
        result = {
            'status': 'error',
            'message': str(e)
        }
    
    return jsonify(result)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 