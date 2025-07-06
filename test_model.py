import pickle
import pandas as pd

# Test the model loading
try:
    with open('patel.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
    
    # Test with sample data
    sample_data = {
        'gender': 0,  # Female
        'SeniorCitizen': 0,
        'Partner': 1,  # Yes
        'Dependents': 0,  # No
        'tenure': 12,
        'PhoneService': 1,  # Yes
        'MultipleLines': 1,  # No
        'InternetService': 1,  # DSL
        'OnlineSecurity': 1,  # No
        'OnlineBackup': 1,  # No
        'DeviceProtection': 1,  # No
        'TechSupport': 1,  # No
        'StreamingTV': 1,  # No
        'StreamingMovies': 1,  # No
        'Contract': 0,  # Month-to-month
        'PaperlessBilling': 1,  # Yes
        'PaymentMethod': 0,  # Electronic check
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    input_df = pd.DataFrame([sample_data])
    input_df = input_df.astype(float)
    
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability > 0.5)
    
    print(f"✅ Test prediction successful!")
    print(f"   Probability: {probability:.2%}")
    print(f"   Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    
except Exception as e:
    print(f"❌ Error: {e}") 