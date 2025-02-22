from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Trained Data/real_estate_sales_model.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Helper functions for safe conversions
        def safe_float(value, default=0.0):
            try:
                return float(value) if value.strip() else default
            except ValueError:
                return default

        def safe_int(value, default=0):
            try:
                return int(value) if value.strip() else default
            except ValueError:
                return default

        # Extract values safely
        data = {
            'Assessed Value': safe_float(request.form.get('assessed_value', '0')),
            'Sales Ratio': safe_float(request.form.get('sales_ratio', '0')),
            'Year': safe_int(request.form.get('year', '0')),
            'Month': safe_int(request.form.get('month', '0')),
            
            'Day': safe_int(request.form.get('day', '1')),
        }

        # Convert dictionary to Pandas DataFrame
        new_data = pd.DataFrame([data])

        # Make prediction
        predicted_sales = model.predict(new_data)[0]

        # Render result page with prediction
        return render_template('result.html', prediction=f"Predicted Sales: {predicted_sales:.2f}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
