import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("student_data.csv")

# -------------------------------
# TITLE
# -------------------------------

st.title("🎓 Student Performance Dashboard")
st.write("Data Analysis + Machine Learning Prediction System")

# -------------------------------
# DATA PREVIEW
# -------------------------------

st.header("📂 Dataset Preview")

st.dataframe(df.head())

# -------------------------------
# KEY METRICS
# -------------------------------

st.header("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Students",
        len(df)
    )

with col2:
    st.metric(
        "Average Math Score",
        round(df['math score'].mean(), 2)
    )

with col3:
    st.metric(
        "Average Reading Score",
        round(df['reading score'].mean(), 2)
    )

# -------------------------------
# SIDEBAR FILTER
# -------------------------------

st.sidebar.header("🔍 Filter Data")

selected_gender = st.sidebar.selectbox(
    "Select Gender",
    df['gender'].unique()
)

filtered_df = df[df['gender'] == selected_gender]

st.subheader(f"Filtered Data ({selected_gender})")

st.dataframe(filtered_df.head())

# -------------------------------
# VISUALIZATIONS
# -------------------------------

st.header("📈 Visualizations")

# HISTOGRAM
st.subheader("Math Score Distribution")

fig1, ax1 = plt.subplots(figsize=(6,4))

sns.histplot(
    df['math score'],
    kde=True,
    ax=ax1
)

st.pyplot(fig1)

# BAR CHART
st.subheader("Average Math Scores by Gender")

fig2, ax2 = plt.subplots(figsize=(6,4))

sns.barplot(
    x='gender',
    y='math score',
    data=df,
    ax=ax2
)

st.pyplot(fig2)

# SCATTER PLOT
st.subheader("Reading Score vs Writing Score")

fig3, ax3 = plt.subplots(figsize=(6,4))

sns.scatterplot(
    x='reading score',
    y='writing score',
    hue='gender',
    data=df,
    ax=ax3
)

st.pyplot(fig3)

# HEATMAP
st.subheader("Correlation Heatmap")

fig4, ax4 = plt.subplots(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    ax=ax4
)

st.pyplot(fig4)

# -------------------------------
# MACHINE LEARNING MODEL
# -------------------------------

st.header("🤖 Machine Learning Model")

# FEATURES & TARGET
X = df[['reading score', 'writing score']]
y = df['math score']

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL
model = LinearRegression()

model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# EVALUATION
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Model Performance")

st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# -------------------------------
# PREDICTION SYSTEM
# -------------------------------

st.header("🔥 Predict Student Math Score")

reading_input = st.slider(
    "Enter Reading Score",
    0,
    100,
    50
)

writing_input = st.slider(
    "Enter Writing Score",
    0,
    100,
    50
)

# PREDICT BUTTON
if st.button("Predict Math Score"):

    input_data = np.array([
        [reading_input, writing_input]
    ])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Math Score: {prediction[0]:.2f}"
    )

# -------------------------------
# FOOTER
# -------------------------------

st.write("---")
st.write("Built using Streamlit, Pandas, Seaborn & Scikit-learn")