import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.title("Model Comparison")

st.info(
    "This page compares all tested models and helps identify which one performed best."
)

df = pd.read_csv("StudentsPerformance.csv")

df["average_score"] = (
    df["math score"] + df["reading score"] + df["writing score"]
) / 3

X = df.drop(columns=["math score", "reading score", "writing score", "average_score"])
y = df["average_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=8),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results.append({
        "Model": name,
        "MSE": round(mean_squared_error(y_test, preds), 4),
        "R2": round(r2_score(y_test, preds), 4)
    })

results_df = pd.DataFrame(results)

st.subheader("Comparison Table")
st.dataframe(results_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    metric = st.selectbox("Choose metric to compare", ["R2", "MSE"])

with col2:
    selected_model = st.selectbox("Inspect one model", results_df["Model"].tolist())

fig = px.bar(
    results_df,
    x="Model",
    y=metric,
    color="Model",
    text=metric,
    title=f"{metric} Comparison Across Models"
)
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

if metric == "R2":
    best_model = results_df.loc[results_df["R2"].idxmax(), "Model"]
    best_value = results_df["R2"].max()
    st.success(f"Best model by R²: {best_model} ({best_value:.4f})")
else:
    best_model = results_df.loc[results_df["MSE"].idxmin(), "Model"]
    best_value = results_df["MSE"].min()
    st.success(f"Best model by MSE: {best_model} ({best_value:.4f})")

st.subheader("Selected Model Details")
selected_row = results_df[results_df["Model"] == selected_model]
st.dataframe(selected_row, use_container_width=True)

st.subheader("Quick Interpretation")

if selected_model == "Linear Regression":
    st.write("Linear Regression gives a strong baseline and performed best in this project.")
elif selected_model == "Decision Tree":
    st.write("Decision Tree is easier to interpret, but it performed worse on unseen data.")
elif selected_model == "Random Forest":
    st.write("Random Forest captures more complex patterns, but did not outperform the best regression models here.")
elif selected_model == "Ridge Regression":
    st.write("Ridge Regression performed very similarly to Linear Regression and adds regularization.")
elif selected_model == "Lasso Regression":
    st.write("Lasso Regression also performed reasonably well and can reduce the influence of weaker features.")

st.markdown("""
### What this comparison shows

- Some models perform better than others on this dataset.
- Regression-based models outperformed the tree-based models in this project.
- Demographic and background variables explain part of student performance, but not all of it.
""")
