import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set the page config
st.set_page_config(page_title="K-Means Clustering", layout="centered")

# Set title
st.title("K-Means Clustering Visualizer by Tanananya Thongkum")

# Display Dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
for center in centroids:
    circle = Circle(center, radius=0.6, color='gray', alpha=0.2, linestyle='--', linewidth=2, fill=True)
    ax.add_patch(circle)
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
