import plotly.express as px
import pandas as pd

def plot_cluster_scatter(coords, labels, filenames):
    cluster_df = pd.DataFrame({
        "Document": filenames,
        "Cluster": [str(lbl) for lbl in labels],
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1]
    })
    
    fig = px.scatter(
        cluster_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster', 
        hover_name='Document', 
        title="Cluster Visualization (2D PCA)"
    )
    fig.update_traces(marker=dict(size=20, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    return fig
