import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Predict")

st.info(
    "This page allows the user to enter student background information "
    "and predict the student's average score using the trained model."
)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("StudentsPerformance.csv")

df["average_score"] = (
    df["math score"] + df["reading score"] + df["writing score"]
) / 3

# -------------------------
# Prepare training data
# -------------------------
X = df.drop(columns=["math score", "reading score", "writing score", "average_score"])
y = df["average_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_encoded = pd.get_dummies(X_train, drop_first=True)

# -------------------------
# Train model
# -------------------------
model = LinearRegression()
model.fit(X_train_encoded, y_train)

# -------------------------
# User input
# -------------------------
st.subheader("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox(
        "Race/Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"]
    )
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

with col2:
    parental_education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]
    )

    test_preparation = st.selectbox(
        "Test Preparation Course",
        ["none", "completed"]
    )

# -------------------------
# Build input dataframe
# -------------------------
input_df = pd.DataFrame([{
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": parental_education,
    "lunch": lunch,
    "test preparation course": test_preparation
}])

input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Average Score"):
    predicted_score = model.predict(input_encoded)[0]

    st.success(f"Predicted Average Score: {predicted_score:.2f}")

    if predicted_score >= 80:
        st.write("This profile is associated with relatively high predicted performance.")
    elif predicted_score >= 60:
        st.write("This profile is associated with moderate predicted performance.")
    else:
        st.write("This profile is associated with lower predicted performance.")

# -------------------------
# Explanation
# -------------------------
with st.expander("How does this page work?"):
    st.markdown("""
This page uses the trained Linear Regression model from the project.

The model takes student background variables such as:

- gender
- race/ethnicity
- parental education
- lunch type
- test preparation course

and predicts the student's **average_score**.
""")
