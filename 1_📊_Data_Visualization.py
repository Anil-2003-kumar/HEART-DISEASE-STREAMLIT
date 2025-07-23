import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Heart Disease Data Visualizations")

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

st.subheader("📈 Age Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df["age"], bins=20, kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("🩺 Heart Disease Count")
st.bar_chart(df["target"].value_counts())

st.subheader("💓 Cholesterol vs Heart Disease")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x="target", y="chol", ax=ax2)
st.pyplot(fig2)

st.subheader("📉 Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)
