import streamlit as st

st.title("Insights")

st.info(
    "This page summarizes the most important findings from the EDA, preprocessing, and modeling stages."
)

st.subheader("Key Findings from EDA")
st.markdown("""
- Math, reading, and writing scores are strongly related to each other.
- Students who completed the test preparation course generally performed better.
- Parent education background shows visible differences in writing and average score.
- Reading and writing scores have a strong positive relationship.
""")

st.subheader("What Preprocessing Helped With")
st.markdown("""
- A single target variable, **average_score**, was created to summarize performance.
- Categorical student background variables were encoded into numeric form.
- Train/test split allowed model evaluation on unseen data.
""")

st.subheader("Modeling Insights")
st.markdown("""
- Five models were tested: Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, and Random Forest.
- Linear-style regression models performed better than Decision Tree and Random Forest on this dataset.
- This suggests that the relationship between the available student features and average score may be better captured by simpler regression-based approaches here.
""")

st.subheader("Most Important Interpretation")
st.markdown("""
The results show that student performance is influenced by several educational and demographic factors, but these variables alone are not enough to perfectly predict outcomes. This means there are likely other important influences not included in the dataset.
""")

st.subheader("Real-World Relevance")
st.markdown("""
This project can help with:

- identifying student groups that may benefit from additional support
- understanding how educational background and preparation connect to outcomes
- informing future educational intervention strategies
""")

st.subheader("Limitations")
st.markdown("""
- The dataset contains only a limited set of background variables.
- Important factors such as motivation, attendance, study habits, and school environment are not included.
- Because of this, model performance remains moderate rather than highly predictive.
""")

st.subheader("Future Improvements")
st.markdown("""
Possible next steps for this project:

- try additional models and hyperparameter tuning
- add more features if available
- improve the dashboard with more user controls
- include deeper interpretation of coefficients and feature importance
- test whether separate subject scores can be modeled more effectively than one combined average score
""")

st.success("This concludes the student performance analysis dashboard.")