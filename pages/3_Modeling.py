import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.title("Modeling")

st.info(
    "This section lets you explore how student background factors relate to predicted average score, "
    "and how model settings affect performance."
)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("StudentsPerformance.csv")

df["average_score"] = (
    df["math score"] + df["reading score"] + df["writing score"]
) / 3

# Human-friendly renamed column for UI
df["income_level"] = df["lunch"].map({
    "standard": "Higher income family",
    "free/reduced": "Low income family"
})

X = df.drop(columns=["math score", "reading score", "writing score", "average_score", "lunch"])
X["income_level"] = df["income_level"]
y = df["average_score"]

# -------------------------
# Main controls (human-friendly)
# -------------------------
st.subheader("Choose a Student Profile")

col1, col2 = st.columns(2)

with col1:
    model_choice = st.selectbox(
        "Select a model",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    gender_label = st.selectbox(
        "Student gender",
        ["Female student", "Male student"]
    )

    income_label = st.selectbox(
        "Family income level",
        ["Higher income family", "Low income family"]
    )

with col2:
    test_prep_label = st.selectbox(
        "Test preparation",
        ["Completed test preparation course", "Did not complete test preparation course"]
    )

    parent_edu_label = st.selectbox(
        "Parent education background",
        [
            "Parents had higher education (bachelor's degree)",
            "Parents had higher education (master's degree)",
            "Parents attended some college",
            "Parents had associate degree",
            "Parents completed high school",
            "Parents did not complete high school"
        ]
    )

    race_label = st.selectbox(
        "Race/ethnicity group",
        ["Group A", "Group B", "Group C", "Group D", "Group E"]
    )

# -------------------------
# Map labels to dataset values
# -------------------------
gender_map = {
    "Female student": "female",
    "Male student": "male"
}

income_map = {
    "Higher income family": "Higher income family",
    "Low income family": "Low income family"
}

test_prep_map = {
    "Completed test preparation course": "completed",
    "Did not complete test preparation course": "none"
}

parent_edu_map = {
    "Parents had higher education (bachelor's degree)": "bachelor's degree",
    "Parents had higher education (master's degree)": "master's degree",
    "Parents attended some college": "some college",
    "Parents had associate degree": "associate's degree",
    "Parents completed high school": "high school",
    "Parents did not complete high school": "some high school"
}

race_map = {
    "Group A": "group A",
    "Group B": "group B",
    "Group C": "group C",
    "Group D": "group D",
    "Group E": "group E"
}

selected_gender = gender_map[gender_label]
selected_income = income_map[income_label]
selected_test_prep = test_prep_map[test_prep_label]
selected_parent_edu = parent_edu_map[parent_edu_label]
selected_race = race_map[race_label]

# -------------------------
# Default advanced settings
# -------------------------
test_size = 0.2
random_state = 42
alpha = None
max_depth = None
min_samples_split = None
n_estimators = None

# -------------------------
# Advanced settings (optional)
# -------------------------
with st.expander("Advanced model settings (optional)"):
    test_size = st.slider(
        "How much data to reserve for testing",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    random_state = st.slider(
        "Random seed",
        min_value=1,
        max_value=100,
        value=42,
        step=1
    )

    if model_choice == "Ridge Regression":
        alpha = st.slider("Regularization strength (Ridge alpha)", 0.01, 10.0, 1.0, 0.01)

    elif model_choice == "Lasso Regression":
        alpha = st.slider("Regularization strength (Lasso alpha)", 0.001, 1.0, 0.1, 0.001)

    elif model_choice == "Decision Tree":
        max_depth = st.slider("Tree depth", 1, 20, 5, 1)
        min_samples_split = st.slider("Minimum samples required to split", 2, 20, 2, 1)

    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of trees", 10, 300, 100, 10)
        max_depth = st.slider("Maximum tree depth", 1, 20, 8, 1)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)

# -------------------------
# Build selected model
# -------------------------
if model_choice == "Linear Regression":
    model = LinearRegression()

elif model_choice == "Ridge Regression":
    model = Ridge(alpha=alpha if alpha is not None else 1.0)

elif model_choice == "Lasso Regression":
    model = Lasso(alpha=alpha if alpha is not None else 0.1)

elif model_choice == "Decision Tree":
    model = DecisionTreeRegressor(
        max_depth=max_depth if max_depth is not None else 5,
        min_samples_split=min_samples_split if min_samples_split is not None else 2,
        random_state=random_state
    )

elif model_choice == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=n_estimators if n_estimators is not None else 100,
        max_depth=max_depth if max_depth is not None else 8,
        random_state=random_state
    )

# -------------------------
# Fit and evaluate selected model
# -------------------------
model.fit(X_train_encoded, y_train)
y_pred = model.predict(X_test_encoded)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------
# Predict chosen profile
# -------------------------
profile_df = pd.DataFrame([{
    "gender": selected_gender,
    "race/ethnicity": selected_race,
    "parental level of education": selected_parent_edu,
    "test preparation course": selected_test_prep,
    "income_level": selected_income
}])

profile_encoded = pd.get_dummies(profile_df, drop_first=True)
profile_encoded = profile_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

predicted_score = model.predict(profile_encoded)[0]

# -------------------------
# Show profile summary
# -------------------------
st.subheader("Selected Student Profile")
st.markdown(f"""
- **Gender:** {gender_label}  
- **Family income level:** {income_label}  
- **Test preparation:** {test_prep_label}  
- **Parent education background:** {parent_edu_label}  
- **Race/ethnicity:** {race_label}  
""")

# -------------------------
# Main metrics
# -------------------------
st.subheader("Prediction and Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Predicted average score", f"{predicted_score:.2f}")

with m2:
    st.metric("MSE", f"{mse:.4f}")

with m3:
    st.metric("R²", f"{r2:.4f}")

# -------------------------
# Split summary
# -------------------------
st.subheader("Training/Test Split")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Training rows", X_train.shape[0])
with c2:
    st.metric("Testing rows", X_test.shape[0])
with c3:
    st.metric("Training target rows", y_train.shape[0])
with c4:
    st.metric("Testing target rows", y_test.shape[0])

# -------------------------
# Explain current settings
# -------------------------
st.subheader("Current Model Setup")
st.write(f"**Model used:** {model_choice}")
st.write(f"**Test split:** {test_size}")
st.write(f"**Random seed:** {random_state}")

if alpha is not None:
    st.write(f"**Regularization strength:** {alpha}")
if max_depth is not None:
    st.write(f"**Tree depth:** {max_depth}")
if min_samples_split is not None:
    st.write(f"**Minimum samples required to split:** {min_samples_split}")
if n_estimators is not None:
    st.write(f"**Number of trees:** {n_estimators}")

# -------------------------
# Quick interpretation
# -------------------------
if r2 > 0.2:
    st.success("This configuration gives relatively better predictive performance.")
elif r2 > 0:
    st.info("This configuration captures some signal, but performance is still limited.")
else:
    st.warning("This configuration performs poorly on the test set.")

# -------------------------
# Benchmark table across models
# -------------------------
comparison_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=8)
}

results = []

for name, comp_model in comparison_models.items():
    comp_model.fit(X_train_encoded, y_train)
    comp_pred = comp_model.predict(X_test_encoded)

    results.append({
        "Model": name,
        "MSE": round(mean_squared_error(y_test, comp_pred), 4),
        "R2": round(r2_score(y_test, comp_pred), 4)
    })

results_df = pd.DataFrame(results)

best_r2_model = results_df.loc[results_df["R2"].idxmax(), "Model"]
best_r2_value = results_df["R2"].max()
best_mse_model = results_df.loc[results_df["MSE"].idxmin(), "Model"]
best_mse_value = results_df["MSE"].min()

st.subheader("Model Results Comparison")
st.dataframe(results_df, use_container_width=True)

b1, b2 = st.columns(2)
with b1:
    st.success(f"Best R²: {best_r2_model} ({best_r2_value:.4f})")
with b2:
    st.success(f"Lowest MSE: {best_mse_model} ({best_mse_value:.4f})")

st.subheader("Model Notes")
if model_choice == "Linear Regression":
    st.write("Linear Regression is a simple baseline model.")
elif model_choice == "Ridge Regression":
    st.write("Ridge Regression adds regularization and can improve stability.")
elif model_choice == "Lasso Regression":
    st.write("Lasso Regression can reduce the influence of weaker features.")
elif model_choice == "Decision Tree":
    st.write("Decision Tree is easier to interpret, but may overfit depending on depth.")
elif model_choice == "Random Forest":
    st.write("Random Forest combines many trees and can capture more complex patterns.")
