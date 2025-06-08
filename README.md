# customer_segmentation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("Customer Segmentation using K-Means Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data.head())

    # Selecting features
    features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # User input for number of clusters
    k = st.slider("Select number of clusters (k)", 2, 10, 5)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    data['Cluster'] = cluster_labels

    # Plot
    st.subheader("Cluster Plot")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette="Set1")
    plt.title(f'Customer Segments (k={k})')
    st.pyplot(plt)
