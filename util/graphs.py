import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_genre_cooccurrence_bo(df, box_office_col='Worldwide_Corrected', min_count=5):
    df = df.copy()
    df['Genre_list'] = df['Genres'].str.split(',').apply(lambda x: [g.strip() for g in x])
    all_genres = sorted(set(g for sublist in df['Genre_list'] for g in sublist))

    freq_matrix = pd.DataFrame(0, index=all_genres, columns=all_genres)
    bo_matrix = pd.DataFrame(0.0, index=all_genres, columns=all_genres)

    for genres, bo in zip(df['Genre_list'], df[box_office_col]):
        unique_genres = sorted(set(genres))
        for g in unique_genres:
            freq_matrix.loc[g, g] += 1
            bo_matrix.loc[g, g] += bo
        for i, g1 in enumerate(unique_genres):
            for g2 in unique_genres[i+1:]:
                freq_matrix.loc[g1, g2] += 1
                freq_matrix.loc[g2, g1] += 1
                bo_matrix.loc[g1, g2] += bo
                bo_matrix.loc[g2, g1] += bo

    avg_bo_matrix = bo_matrix / freq_matrix.replace(0, np.nan)
    mask = freq_matrix < min_count

    return freq_matrix, avg_bo_matrix, mask


def plot_heatmap_with_annotations(avg_bo, freq, mask):
    z = avg_bo.fillna(0).values
    genres = avg_bo.index.tolist()

    annotations = []
    for i, row_genre in enumerate(genres):
        for j, col_genre in enumerate(genres):
            if mask.iloc[i, j]:
                text = ""
            else:
                text = f"{freq.iloc[i, j]:.0f}"
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(color='white' if z[i,j] > np.nanmax(z)/2 else 'black', size=10)
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=genres,
        y=genres,
        colorscale='hot',
        colorbar=dict(title="Avg Box Office ($)"),
        hovertemplate='<b>%{y} + %{x}</b><br>Avg Box Office: %{z:$,.0f}<br>Frequency: %{text}<extra></extra>',
        text=freq.values.astype(str)
    ))

    fig.update_layout(
        title="Genre Pair Average Box Office Heatmap with Frequency",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange='reversed'),
        annotations=annotations,
        height=800,
        width=900,
        template='plotly_dark'
    )
    return fig
