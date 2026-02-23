import plotly.express as px
import pandas as pd

def plot_similarity_heatmap(similarity_matrix, filenames):
    # Truncate to max 25 chars for axes
    truncated_names = [name[:22] + "..." if len(name) > 25 else name for name in filenames]
    
    sim_df = pd.DataFrame(similarity_matrix, index=truncated_names, columns=truncated_names)
    
    fig_sim = px.imshow(
        sim_df, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r"
    )
    
    # We want to show full filenames on hover using customdata
    full_names_grid = [[f"Row: {fy}<br>Col: {fx}" for fx in filenames] for fy in filenames]
    fig_sim.update_traces(
        customdata=full_names_grid,
        hovertemplate="%{customdata}<br>Similarity: %{z:.2f}<extra></extra>"
    )

    return fig_sim
