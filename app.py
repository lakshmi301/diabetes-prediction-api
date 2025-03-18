from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  
import numpy as np
import shap

# Load the model and scaler
model = joblib.load('diabetes_voting_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# ✅ API home route
@app.route('/', methods=['GET'])
def home():
    return "Diabetes Prediction API is Running!"

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """Predicts diabetes based on input features."""
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Invalid input format. Missing 'features' key."}), 400

        features = data["features"]

        required_fields = [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ]

        missing_fields = [field for field in required_fields if field not in features]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Convert input to numpy array
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

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        # Generate AI insights
        insights = generate_insights(features)

        return jsonify({
            "prediction": result,
            "insights": insights
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ SHAP feature importance route
@app.route('/explain', methods=['POST'])
def explain():
    """Returns SHAP feature importance for a prediction."""
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Invalid input format. Missing 'features' key."}), 400

        features = data["features"]

        # Prepare the input data
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

        # Generate SHAP feature importance
        explainer = shap.Explainer(model, input_data_scaled)
        shap_values = explainer(input_data_scaled)

        # Extract feature importance values
        feature_importance = {}
        for i, feature in enumerate(features.keys()):
            feature_importance[feature] = shap_values.values[0][i]

        return jsonify({
            "feature_importance": feature_importance
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ AI Insights Function
def generate_insights(features):
    """Generates AI-based insights based on the input features."""
    insights = []

    # Glucose insights
    if float(features["Glucose"]) > 140:
        insights.append("High glucose levels detected. Consider dietary modifications and regular check-ups.")
    elif float(features["Glucose"]) < 70:
        insights.append("Low glucose levels detected. Consult your healthcare provider.")

    # BMI insights
    bmi = float(features["BMI"])
    if bmi >= 30:
        insights.append("High BMI detected. Maintain a healthy diet and engage in regular physical activity.")
    elif bmi < 18.5:
        insights.append("Low BMI detected. You may need to increase your caloric intake.")

    # Age insights
    age = int(features["Age"])
    if age > 45:
        insights.append("Age is a significant factor for diabetes risk. Regular screenings recommended.")

    # Insulin insights
    insulin = float(features["Insulin"])
    if insulin > 200:
        insights.append("High insulin levels detected. Monitor your sugar intake and consult a physician.")
    elif insulin < 50:
        insights.append("Low insulin levels detected. This may indicate potential hypoglycemia.")

    if not insights:
        insights.append("No significant risk factors detected based on the provided data.")

    return insights


# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
