import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="Habit Tracker AI", page_icon="🏃‍♂️")

# --- TITLE & DESCRIPTION ---
st.title("🌱 Habit Quality Score Predictor")
st.markdown("""
Predict your **Habit Quality Score** based on your daily routines. 
This app uses a Random Forest model trained on your habit dataset.
""")

# --- DATA LOADING & MODEL TRAINING ---
# Based on your project_1.ipynb logic
@st.cache_resource
def train_model():
    # Load your dataset (Replace with your actual CSV path)
    df = pd.read_csv("habit_dataset.csv")
    
    # Preprocessing: Encoding categorical columns as done in your notebook
    # We create a mapping to handle categorical inputs
    categorical_cols = [
        "bed_time_period", "sleep_consistency", "late_night_screen", 
        "app_usage_type", "diet_type", "meal_timings_regular", 
        "exercise_type", "activity_intensity", "occupation_type", 
        "work_environment", "reward_indulgence"
    ]
    
    # Convert categorical to codes (simple label encoding for RF)
    df_encoded = df.copy()
    mappings = {}
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].astype('category')
        mappings[col] = dict(enumerate(df_encoded[col].cat.categories))
        df_encoded[col] = df_encoded[col].cat.codes

    X = df_encoded.drop(columns=["habit_quality_score"])
    y = df_encoded["habit_quality_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Using Random Forest Regressor as requested
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, mappings, X.columns

model, mappings, feature_columns = train_model()

# --- USER INPUT SIDEBAR ---
st.sidebar.header("Log Your Daily Habits")

def user_input_features():
    # Sleep & Routine
    sleep_hours = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.1)
    sleep_quality = st.sidebar.slider("Sleep Quality (0-10)", 0, 10, 5)
    bed_time = st.sidebar.selectbox("Bed Time Period", ["Early", "Normal", "Late"])
    consistency = st.sidebar.selectbox("Sleep Consistency", ["Regular", "Irregular"])
    nap_mins = st.sidebar.number_input("Nap Minutes", 0, 120, 20)
    disturbances = st.sidebar.slider("Sleep Disturbances", 0, 5, 1)

    # Digital Habits
    screen_time = st.sidebar.slider("Screen Time (Hours)", 1.0, 12.0, 4.0)
    late_screen = st.sidebar.selectbox("Late Night Screen?", ["Yes", "No"])
    app_type = st.sidebar.selectbox("Main App Usage", ["Work", "Social", "Entertainment"])

    # Food & Health
    calories = st.sidebar.number_input("Daily Calories", 1200, 4000, 2000)
    junk_food = st.sidebar.slider("Junk Food Servings", 0, 6, 1)
    healthy_meals = st.sidebar.slider("Healthy Meals Count", 0, 4, 3)
    water = st.sidebar.slider("Water Intake (Liters)", 1.0, 5.0, 2.5)
    diet = st.sidebar.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
    meal_reg = st.sidebar.selectbox("Regular Meal Timings?", ["Yes", "No"])

    # Physical Activity
    phys_mins = st.sidebar.slider("Physical Activity (Mins)", 0, 180, 30)
    ex_type = st.sidebar.selectbox("Exercise Type", ["None", "Cardio", "Strength", "Yoga"])
    intensity = st.sidebar.selectbox("Activity Intensity", ["Low", "Moderate", "High"])

    # Mental & Lifestyle
    stress = st.sidebar.slider("Stress Level (0-10)", 0, 10, 3)
    focus = st.sidebar.slider("Focus Level (0-10)", 0, 10, 7)
    occupation = st.sidebar.selectbox("Occupation", ["Student", "Employee", "Freelancer"])
    outdoor = st.sidebar.number_input("Outdoor Time (Mins)", 0, 240, 60)
    work_env = st.sidebar.selectbox("Work Environment", ["Home", "Office", "Hybrid"])
    streak = st.sidebar.number_input("Habit Streak (Days)", 0, 365, 10)
    reward = st.sidebar.selectbox("Reward Indulgence?", ["Yes", "No"])

    # Store inputs in a dictionary
    data = {
        'sleep_hours': sleep_hours, 'sleep_quality': sleep_quality, 'bed_time_period': bed_time,
        'sleep_consistency': consistency, 'nap_minutes': nap_mins, 'sleep_disturbances': disturbances,
        'screen_time_hours': screen_time, 'late_night_screen': late_screen, 'app_usage_type': app_type,
        'daily_calories': calories, 'junk_food_servings': junk_food, 'healthy_meals_count': healthy_meals,
        'water_intake_liters': water, 'diet_type': diet, 'meal_timings_regular': meal_reg,
        'physical_activity_minutes': phys_mins, 'exercise_type': ex_type, 'activity_intensity': intensity,
        'stress_level': stress, 'focus_level': focus, 'occupation_type': occupation,
        'outdoor_time_minutes': outdoor, 'work_environment': work_env, 'habit_streak_days': streak,
        'reward_indulgence': reward
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- PREDICTION LOGIC ---
# Map user text inputs to the same numeric codes used in training
input_encoded = input_df.copy()
for col, map_dict in mappings.items():
    inv_map = {v: k for k, v in map_dict.items()}
    input_encoded[col] = input_encoded[col].map(inv_map)

# Predict
prediction = model.predict(input_encoded)

# --- DISPLAY RESULTS ---
st.subheader("Results")
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Habit Score", f"{prediction[0]:.2f}/100")

with col2:
    if prediction[0] >= 70:
        st.success("Great job! Your habits are highly productive.")
    elif prediction[0] >= 40:
        st.warning("You're doing okay, but there's room for improvement.")
    else:
        st.error("Consider adjusting your routine for better health.")

# Feature Importance Chart
st.subheader("Model Insights")
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=feature_columns)
st.bar_chart(feat_importances.sort_values(ascending=False).head(10))