from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
xgb_model = joblib.load('XGBoost_model.pkl')

# Route to handle home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        person_age = float(request.form['person_age'])
        person_gender = int(request.form['person_gender'])  # 0 for male, 1 for female
        person_education = int(request.form['person_education'])  # 0 - 4 (label-encoded)
        person_income = float(request.form['person_income'])
        person_emp_exp = int(request.form['person_emp_exp'])
        loan_amnt = float(request.form['loan_amnt'])
        loan_int_rate = float(request.form['loan_int_rate'])
        loan_percent_income = float(request.form['loan_percent_income'])
        cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])
        credit_score = int(request.form['credit_score'])
        
        # One-hot encoded values (home ownership and loan intent)
        home_ownership = request.form['home_ownership']
        loan_intent = request.form['loan_intent']
        
        # Prepare the data for prediction
        custom_test_data = pd.DataFrame([{
            'person_age': person_age,
            'person_gender': person_gender,
            'person_education': person_education,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,

            # One-hot encoding for home ownership
            'person_home_ownership_OTHER': 1 if home_ownership == 'Other' else 0,
            'person_home_ownership_OWN': 1 if home_ownership == 'Own' else 0,
            'person_home_ownership_RENT': 1 if home_ownership == 'Rent' else 0,

            # One-hot encoding for loan intent
            'loan_intent_EDUCATION': 1 if loan_intent == 'Education' else 0,
            'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'Home Improvement' else 0,
            'loan_intent_MEDICAL': 1 if loan_intent == 'Medical' else 0,
            'loan_intent_PERSONAL': 1 if loan_intent == 'Personal' else 0,
            'loan_intent_VENTURE': 1 if loan_intent == 'Venture' else 0
        }])

        # Make prediction
        prediction = xgb_model.predict(custom_test_data)

        # Return prediction result
        if prediction[0] == 1:
            result = 'Approved'
        else:
            result = 'Rejected'
        
        return render_template('index.html', prediction_text=f'Loan Status: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
