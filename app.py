from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('diabetes_voting_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

# Ensure SHAP plot folder exists
if not os.path.exists("static"):
    os.makedirs("static")


# Function to Generate Detailed AI Insights
def generate_insights(features):
    """Generates detailed AI-based insights based on the input features."""
    insights = []

    # Glucose Insights
    glucose = float(features["Glucose"])
    if glucose >= 200:
        insights.append(" **Severely high glucose levels detected.** This may indicate hyperglycemia, a serious condition linked with diabetes. Immediate medical attention is advised.")
    elif glucose >= 140:
        insights.append(" **High glucose levels detected.** This may indicate prediabetes. Dietary modifications, regular exercise, and routine check-ups are recommended.")
    elif glucose <= 70:
        insights.append(" **Low glucose levels detected.** This could be a sign of hypoglycemia. Consider consuming fast-acting carbohydrates and consult your healthcare provider.")
    else:
        insights.append(" **Glucose levels are within the normal range.** No immediate concern detected.")

    # BMI Insights
    bmi = float(features["BMI"])
    if bmi >= 30:
        insights.append(" **Obese BMI detected.** This increases the risk of diabetes, heart disease, and hypertension. Regular physical activity and dietary modifications are recommended.")
    elif bmi >= 25:
        insights.append(" **Overweight BMI detected.** Increased risk of diabetes and cardiovascular issues. Lifestyle changes, such as healthier eating and exercise, are recommended.")
    elif bmi < 18.5:
        insights.append(" **Low BMI detected.** This may indicate undernutrition. Consult a dietitian to evaluate your nutritional intake.")
    else:
        insights.append(" **BMI is within the healthy range.** Maintain your current lifestyle for optimal health.")

    # Blood Pressure Insights
    bp = float(features["BloodPressure"])
    if bp >= 140:
        insights.append(" **High blood pressure detected.** This increases the risk of heart disease and diabetes. Lifestyle changes and regular monitoring are advised.")
    elif bp < 90:
        insights.append(" **Low blood pressure detected.** This may lead to fatigue or dizziness. Ensure proper hydration and salt intake.")
    else:
        insights.append(" **Blood pressure is within the normal range.** No immediate concerns.")

    # Age Insights
    age = int(features["Age"])
    if age >= 60:
        insights.append(" **Age over 60 detected.** This is a risk factor for diabetes. Regular screenings and lifestyle management are highly recommended.")
    elif age >= 45:
        insights.append(" **Moderate risk due to age.** Individuals over 45 have a higher likelihood of developing diabetes. Regular check-ups are advised.")
    else:
        insights.append("**Age is not a significant risk factor.** Continue with a healthy lifestyle.")

    # Insulin Insights
    insulin = float(features["Insulin"])
    if insulin > 200:
        insights.append(" **High insulin levels detected.** This may indicate insulin resistance. Monitor sugar intake and consult your physician.")
    elif insulin < 50:
        insights.append(" **Low insulin levels detected.** This could indicate potential hypoglycemia risk. Monitor glucose intake carefully.")
    else:
        insights.append(" **Insulin levels are within the normal range.** No immediate concerns.")

    # Diabetes Pedigree Insights
    dpf = float(features["DiabetesPedigreeFunction"])
    if dpf >= 1.0:
        insights.append(" **High diabetes pedigree function detected.** This suggests a stronger genetic predisposition to diabetes. Lifestyle changes and regular monitoring are recommended.")
    elif dpf < 0.2:
        insights.append(" **Low genetic risk factor detected.** Minimal hereditary influence on diabetes risk.")
    else:
        insights.append(" **Moderate genetic risk.** Lifestyle factors are still crucial.")

    # Skin Thickness Insights
    skin = float(features["SkinThickness"])
    if skin >= 40:
        insights.append(" **High skin thickness detected.** This may indicate increased fat deposition, a potential indicator of insulin resistance.")
    elif skin < 10:
        insights.append(" **Low skin thickness detected.** This may indicate low fat reserves or possible malnutrition.")
    else:
        insights.append(" **Normal skin thickness detected.** No immediate concern.")

    return insights


#  SHAP Plot Generation Function
def generate_shap_plot(model, input_data_scaled):
    try:
        explainer = shap.Explainer(model.predict, input_data_scaled)
        shap_values = explainer(input_data_scaled)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, input_data_scaled, show=False)

        # Save the SHAP plot
        shap_output_path = "static/shap_plot.png"
        plt.title("SHAP Feature Importance")
        plt.xlabel("Feature Impact on Prediction")
        plt.savefig(shap_output_path)
        plt.close()

        return shap_output_path
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None


#  Home Route (Form)
@app.route("/")
def home():
    return render_template("index.html")


#  Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input features from form
        features = request.form

        # Validate form fields
        required_fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        for field in required_fields:
            if field not in features:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Prepare input data
        input_data = np.array([[  
            float(features["Pregnancies"]),
            float(features["Glucose"]),
            float(features["BloodPressure"]),
            float(features["SkinThickness"]),
            float(features["Insulin"]),
            float(features["BMI"]),
            float(features["DiabetesPedigreeFunction"]),
            float(features["Age"])
        ]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Get prediction and probabilities
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        # Generate detailed AI insights
        insights = generate_insights(features)

        # Generate SHAP plot
        shap_output_path = generate_shap_plot(model, input_data_scaled)

        # Return the detailed response
        return jsonify({
            "prediction": result,
            "probabilities": {
                "Non-Diabetic": f"{probabilities[0]:.4f}",
                "Diabetic": f"{probabilities[1]:.4f}"
            },
            "insights": insights,
            "shap_plot": shap_output_path
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            "error": str(e)
        }), 500


#  Run Flask Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
