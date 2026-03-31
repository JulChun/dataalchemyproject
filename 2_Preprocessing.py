import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("Preprocessing")

st.info(
    "This section explains how the raw student dataset was prepared before modeling."
)

df = pd.read_csv("StudentsPerformance.csv")

# -------------------------
# Feature engineering
# -------------------------
df["average_score"] = (
    df["math score"] + df["reading score"] + df["writing score"]
) / 3

st.subheader("What was added to the dataset?")
st.success(
    "A new target variable called average_score was created by averaging math, reading, and writing scores."
)

# -------------------------
# User controls
# -------------------------
st.subheader("Explore Preprocessing Steps")

col1, col2 = st.columns(2)

with col1:
    show_raw = st.checkbox("Show original dataset preview", value=True)
    show_engineered = st.checkbox("Show dataset with average_score", value=True)

with col2:
    show_encoded = st.checkbox("Show encoded features", value=True)
    show_split = st.checkbox("Show train/test split summary", value=True)

# -------------------------
# Raw data
# -------------------------
if show_raw:
    st.subheader("Original Data")
    st.dataframe(df.drop(columns=["average_score"]).head(), use_container_width=True)

# -------------------------
# Engineered feature view
# -------------------------
if show_engineered:
    st.subheader("Dataset After Feature Engineering")
    st.dataframe(df.head(), use_container_width=True)

# -------------------------
# Prepare features
# -------------------------
X = df.drop(columns=["average_score"])
y = df["average_score"]

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# -------------------------
# Encoded features
# -------------------------
if show_encoded:
    st.subheader("Encoded Features")
    st.write(
        "Categorical variables were converted into numeric format using one-hot encoding."
    )
    st.dataframe(X_encoded.head(), use_container_width=True)

# -------------------------
# Split summary
# -------------------------
if show_split:
    st.subheader("Train/Test Split Summary")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Training rows", X_train.shape[0])
    with m2:
        st.metric("Testing rows", X_test.shape[0])
    with m3:
        st.metric("Training target rows", y_train.shape[0])
    with m4:
        st.metric("Testing target rows", y_test.shape[0])

# -------------------------
# Explanation section
# -------------------------
with st.expander("Why were these preprocessing steps needed?"):
    st.markdown("""
- **Feature engineering** was used to create one overall performance target: `average_score`.
- **One-hot encoding** was needed because machine learning models require numeric input.
- **Train/test split** was used to evaluate model performance on unseen data.
""")