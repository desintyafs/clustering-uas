import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_excel('output_cluster.xlsx')
df = df.drop(['Unnamed: 0','Clusters'],axis=1)

# Header Interface
st.header("Isi dataset")
st.write(df)

# Create the slider for selecting the number of clusters (K)
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster: ", 2, 10, 1, 1)

# Scatter plot function
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(df)
    df['Labels'] = kmean.labels_

    st.subheader('Cluster Plot')
    fig, axes = plt.subplots(3,1, figsize=(20, 40))

    sns.scatterplot(x=df['Income'], y=df['Seniority'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust), ax=axes[0])
    axes[0].set_title('Income vs Seniority')

    sns.scatterplot(x=df['Income'], y=df['Spending'], hue=df['Labels'], palette=sns.color_palette('hls', n_colors=n_clust), ax=axes[1])
    axes[1].set_title('Income vs Spending')

    sns.scatterplot(x=df['Income'], y=df['Spending'], hue=df['Seniority'], ax=axes[2])
    axes[2].set_title('Income vs Spending vs Seniority')

    st.pyplot(fig)

# Clustering process and scatter plot
if st.button("Proses Klasterisasi dan Tampilkan Plot"):
    clusters = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i).fit(df)
        clusters.append(km.inertia_)

    # Display the elbow plot
    st.subheader("Mencari Elbow")
    st.line_chart(clusters, use_container_width=True)

    k_means(clust)
