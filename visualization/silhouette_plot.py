import plotly.express as px
import pandas as pd
import streamlit as st

def plot_silhouette_scores(scores_dict):
    if not scores_dict:
        st.info("Not enough data to compute silhouette scores across multiple values of k.")
        return None
        
    # Filter out valid scores (-1 means invalid in our computation logic)
    valid_scores = {k: v for k, v in scores_dict.items() if v >= -1}
    
    if not valid_scores:
         st.info("Could not compute valid silhouette scores for these clusters.")
         return None

    df = pd.DataFrame(list(valid_scores.items()), columns=['Number of Clusters (k)', 'Silhouette Score'])
    
    fig = px.line(
        df, 
        x='Number of Clusters (k)', 
        y='Silhouette Score', 
        markers=True,
        title="Silhouette Score vs. Number of Clusters"
    )
    fig.update_layout(yaxis_range=[-1.05, 1.05])
    return fig
