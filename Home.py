import streamlit as st

st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

st.title("Student Performance Dashboard")

st.write("""
This dashboard presents exploratory data analysis and predictive modeling
for the Students Performance dataset.
""")

st.success("Use the sidebar to navigate through different stages of the project.")

st.write("Use the navigation menu on the left to explore:")

st.markdown("""
- EDA  
- Preprocessing  
- Modeling  
- Model Comparison  
- Insights  
""")