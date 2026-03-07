from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_web', methods=['POST'])
def predict_datapoint():
    try:
    
        inputs = request.form.to_dict()

        data = CustomData(
            age=int(inputs['age']),
            gender=inputs['gender'],
            education_level=inputs['education_level'],
            annual_income=float(inputs['annual_income']),
            employment_experience_years=int(inputs['employment_experience_years']),
            home_ownership_status=inputs['home_ownership_status'],
            loan_amount=float(inputs['loan_amount']),
            loan_purpose=inputs['loan_purpose'],
            interest_rate=float(inputs['interest_rate']),
            loan_to_income_ratio=float(inputs['loan_to_income_ratio']),
            credit_history_length_years=int(inputs['credit_history_length_years']),
            credit_score=int(inputs['credit_score']),
            prior_default_flag=int(inputs['prior_default_flag'])
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        pred_class, pred_prob = predict_pipeline.predict(pred_df)

        pred_label = "Accepted" if pred_class[0] == 0 else "Rejected"
        probability_value = f"{pred_prob[0]*100:.2f}%" if pred_prob is not None else "N/A"

        return render_template(
            "home.html",
            prediction=pred_class[0],
            status=pred_label,
            probability=probability_value,
            inputs=inputs
        )

    except Exception as e:
        return render_template("home.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)