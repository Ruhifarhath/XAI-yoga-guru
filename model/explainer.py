import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# --- Load model from disk ---
def load_model():
    return joblib.load("model.pkl")

def clean_lime_feature_name(name):
    if "≤ 0.00" in name:
        return name.replace(" ≤ 0.00", " = No")
    elif "0.00 <" in name and "≤ 1.00" in name:
        return name.replace("0.00 < ", "").replace(" ≤ 1.00", " = Yes")
    return name

def format_weight_line(feature, weight):
    color = "green" if weight > 0 else "red"
    return f"- <span style='color:{color}'>{feature}: {weight:.2f}</span>"

# --- Generate Feature Importance Plot ---
def plot_feature_importance(explanation_obj, feature_names):
    features, weights = zip(*explanation_obj.as_list())
    feature_names = [clean_lime_feature_name(feature) for feature in features]
    
    # Create a DataFrame for better plotting
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": weights
    })
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_df, x="Importance", y="Feature", palette="coolwarm")
    plt.title("Feature Importance for Pose Safety Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt)

# --- Predict + Explain ---
def generate_pose_prediction_and_explanation(pose, conditions, injuries, yoga_level):
    # Prepare user input
    injuries = [i.lower() for i in injuries]
    try:
        yoga_level_score = ["Beginner", "Intermediate", "Advanced"].index(yoga_level) + 1
    except ValueError:
        yoga_level_score = 1

    user_input = {
        'Pose': pose,
        'Age': int(st.session_state.user_data.get("age", 30)),
        'Pregnancy': 1 if "Pregnancy" in conditions else 0,
        'Sciatica': 1 if "Sciatica" in conditions else 0,
        'Herniated Disc': 1 if "Herniated Disc" in conditions else 0,
        'Hypertension': 1 if "Hypertension" in conditions else 0,
        'Arthritis': 1 if "Arthritis" in conditions else 0,
        'Knee Injury': 1 if "knee" in injuries else 0,
        'Yoga Level': yoga_level_score
    }

    # Load model and feature list
    model, feature_names = load_model()

    # Load training data (for LIME explainer fit)
    df_train = pd.read_csv("assets/pose_safety_dataset.csv")
    X_train = pd.get_dummies(df_train.drop(columns=["Pose Safe"]))
    X_train = X_train.reindex(columns=feature_names, fill_value=0)

    # Encode user input to match training features
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    # LIME setup
    categorical_features = [
        i for i, col in enumerate(X_train.columns)
        if "Pose_" in col or "Yoga Level" in col
    ]

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["UNSAFE", "SAFE"],
        mode="classification",
        categorical_features=categorical_features
    )

    # Generate explanation
    explanation_obj = explainer.explain_instance(
        data_row=input_encoded.values[0],
        predict_fn=model.predict_proba,
        num_features=6
    )

    prediction = model.predict(input_encoded)[0]
    prediction_label = "SAFE" if prediction == 1 else "UNSAFE"

    # Natural Language Explanation
    explanation = f"### 🧘 {pose}\n"
    explanation += f"**Prediction**: {prediction_label}\n\n"
    explanation += "**LIME Top Features:**\n"
    for feature, weight in explanation_obj.as_list():
        explanation += f"- {feature}: {weight:.2f}\n"
    
    # Plot Feature Importance
    plot_feature_importance(explanation_obj, feature_names)

    # Adding more natural language explanation
    explanation += "\n**Explanation:**\n"
    if prediction_label == "SAFE":
        explanation += "This pose is considered safe based on your health and injury conditions. The most important factors contributing to this are your lack of hypertension and knee injuries."
    else:
        explanation += "This pose is considered unsafe because it may put additional strain on your joints, especially due to your knee injury or hypertension. The model suggests avoiding this pose."

    return prediction_label, explanation

# --- UI helper (optional) ---
def display_pose_explanation():
    st.title("🧘 AI-Powered Yoga Safety Advisor")

    selected_pose = st.session_state.get("selected_pose", "Child’s Pose")
    conditions = st.session_state.get("user_data", {}).get("medical_conditions", [])
    injuries = st.session_state.get("user_data", {}).get("injuries", [])
    yoga_level = st.session_state.get("user_data", {}).get("yoga_level", "Beginner")

    prediction, explanation = generate_pose_prediction_and_explanation(selected_pose, conditions, injuries, yoga_level)
    st.markdown(explanation)

if __name__ == "__main__":
    display_pose_explanation()
