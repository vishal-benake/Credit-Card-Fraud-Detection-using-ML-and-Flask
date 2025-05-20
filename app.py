from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        input_values = [float(request.form[f'feature{i}']) for i in range(1, 31)]
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]  # Probability of predicted class


        result = "⚠️ Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legitimate Transaction"
        confidence = f"{proba * 100:.2f}%"  # Format as percentage with 2 decimals

        return render_template('result.html', prediction=result, confidence=confidence)
    
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", confidence=None)

if __name__ == '__main__':
    app.run(debug=True)
