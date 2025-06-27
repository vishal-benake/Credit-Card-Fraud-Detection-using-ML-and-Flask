from flask import Flask, request, render_template, session
from src.pipelines.predict_pipeline import CustomData, PredictPipeline
import secrets

application = Flask(__name__)
app = application

# Secure session key
app.secret_key = secrets.token_hex(16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                Time=float(request.form.get("Time")),
                V1=float(request.form.get("V1")),
                V2=float(request.form.get("V2")),
                V3=float(request.form.get("V3")),
                V4=float(request.form.get("V4")),
                V5=float(request.form.get("V5")),
                V6=float(request.form.get("V6")),
                V7=float(request.form.get("V7")),
                V8=float(request.form.get("V8")),
                V9=float(request.form.get("V9")),
                V10=float(request.form.get("V10")),
                V11=float(request.form.get("V11")),
                V12=float(request.form.get("V12")),
                V13=float(request.form.get("V13")),
                V14=float(request.form.get("V14")),
                V15=float(request.form.get("V15")),
                V16=float(request.form.get("V16")),
                V17=float(request.form.get("V17")),
                V18=float(request.form.get("V18")),
                V19=float(request.form.get("V19")),
                V20=float(request.form.get("V20")),
                V21=float(request.form.get("V21")),
                V22=float(request.form.get("V22")),
                V23=float(request.form.get("V23")),
                V24=float(request.form.get("V24")),
                V25=float(request.form.get("V25")),
                V26=float(request.form.get("V26")),
                V27=float(request.form.get("V27")),
                V28=float(request.form.get("V28")),
                Amount=float(request.form.get("Amount"))
            )

            pred_df = data.get_data_as_data_frame()

            pipeline = PredictPipeline()
            pred, proba = pipeline.predict(pred_df)

            prediction_text = "⚠️ Fraudulent Transaction" if pred[0] == 1 else "✅ Legitimate Transaction"

            session['user_input'] = pred_df.to_dict(orient='records')[0]
            session['prediction_text'] = prediction_text
            session['fraud_prob'] = float(proba[0][1])
            session['not_fraud_prob'] = float(proba[0][0])

            return render_template('home.html', prediction_text=prediction_text)

        except Exception as e:
            print("Prediction Error:", e)
            return f"Internal Server Error: {str(e)}", 500

@app.route('/report')
def report():
    user_input = session.get('user_input')
    prediction_text = session.get('prediction_text')
    fraud_prob = session.get('fraud_prob')
    not_fraud_prob = session.get('not_fraud_prob')

    if not user_input or not prediction_text:
        return "Missing session data. Please go back and submit the form again.", 400

    return render_template(
        'report.html',
        prediction_text=prediction_text,
        user_input=user_input,
        fraud_prob=fraud_prob,
        not_fraud_prob=not_fraud_prob
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0")
