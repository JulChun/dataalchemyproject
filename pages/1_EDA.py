import streamlit as st
import pandas as pd
import plotly.express as px

st.title("EDA")

st.info(
    "This section helps explore how student background and preparation relate to academic performance."
)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("StudentsPerformance.csv")

df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

# Human-friendly renamed column for UI
df["income_level"] = df["lunch"].map({
    "standard": "Higher income family",
    "free/reduced": "Low income family"
})

# -------------------------
# Human-friendly filters
# -------------------------
st.subheader("Explore Student Groups")

col1, col2 = st.columns(2)

with col1:
    gender_filter = st.multiselect(
        "Student gender",
        ["Female student", "Male student"],
        default=["Female student", "Male student"]
    )

    income_filter = st.multiselect(
        "Family income level",
        ["Higher income family", "Low income family"],
        default=["Higher income family", "Low income family"]
    )

with col2:
    prep_filter = st.multiselect(
        "Test preparation status",
        ["Completed test preparation course", "Did not complete test preparation course"],
        default=[
            "Completed test preparation course",
            "Did not complete test preparation course"
        ]
    )

    parent_filter = st.multiselect(
        "Parent education background",
        [
            "Parents had higher education (bachelor's degree)",
            "Parents had higher education (master's degree)",
            "Parents attended some college",
            "Parents had associate degree",
            "Parents completed high school",
            "Parents did not complete high school"
        ],
        default=[
            "Parents had higher education (bachelor's degree)",
            "Parents had higher education (master's degree)",
            "Parents attended some college",
            "Parents had associate degree",
            "Parents completed high school",
            "Parents did not complete high school"
        ]
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

prep_map = {
    "Completed test preparation course": "completed",
    "Did not complete test preparation course": "none"
}

parent_map = {
    "Parents had higher education (bachelor's degree)": "bachelor's degree",
    "Parents had higher education (master's degree)": "master's degree",
    "Parents attended some college": "some college",
    "Parents had associate degree": "associate's degree",
    "Parents completed high school": "high school",
    "Parents did not complete high school": "some high school"
}

selected_gender = [gender_map[x] for x in gender_filter]
selected_income = [income_map[x] for x in income_filter]
selected_prep = [prep_map[x] for x in prep_filter]
selected_parent = [parent_map[x] for x in parent_filter]

filtered_df = df[
    (df["gender"].isin(selected_gender)) &
    (df["income_level"].isin(selected_income)) &
    (df["test preparation course"].isin(selected_prep)) &
    (df["parental level of education"].isin(selected_parent))
]

# -------------------------
# Quick overview metrics
# -------------------------
st.subheader("Quick Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Students shown", len(filtered_df))
with m2:
    st.metric("Average math score", f"{filtered_df['math score'].mean():.2f}")
with m3:
    st.metric("Average reading score", f"{filtered_df['reading score'].mean():.2f}")
with m4:
    st.metric("Average writing score", f"{filtered_df['writing score'].mean():.2f}")

# -------------------------
# Dataset preview
# -------------------------
with st.expander("Show filtered dataset preview"):
    preview_df = filtered_df[[
        "gender",
        "race/ethnicity",
        "parental level of education",
        "income_level",
        "test preparation course",
        "math score",
        "reading score",
        "writing score",
        "average_score"
    ]]
    st.dataframe(preview_df, use_container_width=True)

# -------------------------
# Distribution chart
# -------------------------
st.subheader("Score Distribution")

score_choice = st.selectbox(
    "Choose which score to explore",
    ["math score", "reading score", "writing score", "average_score"]
)

fig1 = px.histogram(
    filtered_df,
    x=score_choice,
    nbins=20,
    title=f"Distribution of {score_choice}"
)
st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "This chart shows how student scores are distributed."
)

# -------------------------
# Group comparison
# -------------------------
st.subheader("Compare Student Groups")

group_choice = st.selectbox(
    "Compare groups by",
    [
        "gender",
        "income_level",
        "test preparation course",
        "parental level of education"
    ]
)

metric_choice = st.selectbox(
    "Score to compare",
    ["math score", "reading score", "writing score", "average_score"]
)

fig2 = px.box(
    filtered_df,
    x=group_choice,
    y=metric_choice,
    color=group_choice,
    title=f"{metric_choice} by {group_choice}"
)
fig2.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "This box plot compares score patterns across student groups."
)

# -------------------------
# Interactive relationship chart
# -------------------------
st.subheader("Explore Relationships Between Scores")

x_axis = st.selectbox(
    "Choose X-axis",
    ["math score", "reading score", "writing score", "average_score"],
    index=0,
    key="x_axis"
)

y_axis = st.selectbox(
    "Choose Y-axis",
    ["reading score", "writing score", "math score", "average_score"],
    index=0,
    key="y_axis"
)

color_choice = st.selectbox(
    "Color points by",
    [
        "gender",
        "income_level",
        "test preparation course",
        "race/ethnicity"
    ],
    key="color_choice"
)

fig3 = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color=color_choice,
    title=f"{y_axis} vs {x_axis}"
)
st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "This scatter plot shows the relationship between two score variables and highlights group differences."
)

# -------------------------
# Correlation heatmap
# -------------------------
st.subheader("Correlation Heatmap")

corr = filtered_df[
    ["math score", "reading score", "writing score", "average_score"]
].corr()

fig4 = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Between Scores"
)
st.plotly_chart(fig4, use_container_width=True)

st.caption(
    "This heatmap shows how strongly the score variables are related to each other."
)

# -------------------------
# Main takeaway
# -------------------------
st.subheader("EDA Takeaway")

if len(filtered_df) > 0:
    st.success(
        "The filtered view shows how preparation, family income level, gender, and parent education may relate to student performance."
    )
else:
    st.warning("No students match the selected filters. Try broadening your selections.")
