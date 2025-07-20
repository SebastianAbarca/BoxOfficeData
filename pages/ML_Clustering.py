import streamlit as st
import pandas as pd
import plotly.express as px
from util import clustering # Assuming util/clustering.py is in your util directory


@st.cache_data
def load_data():
    df_final = pd.read_csv('data/final_box_office.csv')
    df_yearly_totals = pd.read_csv('data/yearly_top10_summary.csv')
    df_genre_bridge = pd.read_csv('data/genre_bridge.csv')
    df_production_bridge = pd.read_csv('data/production_bridge.csv')

    # Ensure Year is int for filtering
    df_final['Year'] = df_final['Year'].astype(int)
    df_yearly_totals['Year'] = df_yearly_totals['Year'].astype(int)
    df_genre_bridge['Year'] = df_genre_bridge['Year'].astype(int)
    df_production_bridge['Year'] = df_production_bridge['Year'].astype(int)

    # Sort yearly totals for correct latest year picking
    df_yearly_totals = df_yearly_totals.sort_values('Year').reset_index(drop=True)

    # --- Canonicalize Genres (Crucial for consistent clustering and visualization) ---
    import re
    def canonicalize_genre_string(genres_str):
        if pd.isna(genres_str) or genres_str.strip() == '':
            return ''
        # Remove any non-alphanumeric characters (except comma and space), then split and strip
        cleaned_str = re.sub(r'[^\w, ]', '', genres_str)  # Keep only word chars, comma, space
        cleaned_genres = [g.strip() for g in cleaned_str.split(',') if g.strip()]
        return ", ".join(sorted(cleaned_genres))

    df_final['Genres'] = df_final['Genres'].apply(canonicalize_genre_string)
    df_genre_bridge['Genres'] = df_genre_bridge['Genres'].apply(canonicalize_genre_string)
    # --- END Canonicalization ---

    return df_final, df_yearly_totals, df_genre_bridge, df_production_bridge

# Load data specifically for this page
df_final, df_yearly_totals, df_genre_bridge, _ = load_data()

st.set_page_config(
    page_title="Movie Clusters",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Apply Custom CSS (if desired for this page) ---
st.markdown("""
            <style>
            .stApp{
                background-color: #000000; /*Stratos*/
                color: #FFFFFF; /* Default text color for the whole app */
            }
            /* Styling for the H1 Title */
            h1 {
                color: #FFD700; /* Gold color for the title */
                font-size: 3.5em;
                font-weight: bold;
                text-align: center;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                margin-bottom: 30px;
            }
            /* Override specific component text colors to ensure white */
            p { /* For general st.write() text */
                color: #FFFFFF;
            }
            div[data-testid="stMetricLabel"] {
                color: #B0B0B0; /* Slightly softer white for metric labels */
            }
            div[data-testid="stMetricValue"] {
                color: #FFFFFF; /* Pure white for metric values */
            }
            /* Styling for st.container elements - General Style */
            .stContainer {
                background-color: #05007D; /* A slightly lighter blue/purple for containers to stand out from background */
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            /* Specific border for the Title container (corrected color) */
            div[data-testid="stVerticalBlock"] > div:nth-child(1) .stContainer {
                border: 2px solid #FFFFFF; /* White border for the title container */
            }
            /* NEW: Custom styles for specific containers */
            /* You can target containers by their position or add unique identifiers if needed */

            /* Example: Styling the Clustering container */
            div[data-testid="stVerticalBlock"] > div:nth-child(1) .stContainer { /* Adjusted for this page's structure */
                border: 2px dashed #FF6347; /* Tomato dashed border */
                box-shadow: 0 0 20px rgba(255, 99, 71, 0.6); /* Strong red glow */
            }
            .stSlider label {
                color: #FFFFFF;
            }
            .stRadio {
                margin-bottom: 0px;
            }
            </style>
            """
,unsafe_allow_html=True)

# --- Clustering Page Content ---
st.title("🎬 Movie Clusters by Box Office & Genre") # Use st.title for a page
st.write("Group movies with similar box office performance and genre profiles.")

# Year range slider
year_range = st.slider(
    "Select Year Range for Clustering:",
    min_value=int(df_yearly_totals['Year'].min()),
    max_value=int(df_yearly_totals['Year'].max()),
    value=(int(df_yearly_totals['Year'].min()), int(df_yearly_totals['Year'].max())),
    key='clustering_year_range_slider'
)

df_movies_for_clustering = df_final[(df_final['Year'] >= year_range[0]) & (df_final['Year'] <= year_range[1])].copy()

if not df_movies_for_clustering.empty:
    st.subheader("Clustering Configuration")

    # Outlier handling: Winsorization (Isolation Forest removed)
    st.markdown("#### Outlier Handling (Box Office)")
    winsorize_data = st.checkbox(
        "Winsorize Box Office Extremes (Cap top/bottom values)",
        value=True,
        help="Caps extreme box office values (e.g., highest 1% and lowest 1%) to reduce their disproportionate impact on clustering, without removing the movies."
    )
    winsorize_percentile = st.slider( # Renamed for clarity as it's a single percentile value
        "Winsorization Limits (Percentile):",
        min_value=0.01, # 1st percentile
        max_value=0.10, # 10th percentile
        value=0.01, # Default 1st percentile for both ends (meaning 99th and 1st)
        step=0.01,
        disabled=not winsorize_data,
        help="Set the percentile for capping. A value of 0.01 means values below 1st percentile are set to 1st, and values above 99th percentile are set to 99th."
    )

    df_movies_to_cluster = df_movies_for_clustering.copy()
    initial_movie_count = len(df_movies_to_cluster)
    box_office_cols = ['Worldwide_Adjusted', 'Domestic_Adjusted', 'Foreign_Original'] # Confirming Foreign_Original is intended

    # Convert the single slider value into the (lower, upper) tuple expected by winsorize_box_office
    winsorize_limits_tuple = (winsorize_percentile, 1 - winsorize_percentile)

    # Step 1: Winsorization
    if winsorize_data:
        df_movies_to_cluster = clustering.winsorize_box_office(df_movies_to_cluster, box_office_cols, limits=winsorize_limits_tuple)
        st.info(f"Box office data winsorized at {winsorize_percentile*100:.1f}% and {(1-winsorize_percentile)*100:.1f}% percentiles.")


    if len(df_movies_to_cluster) >= 2:
        optimal_k_estimate = clustering.get_best_k(df_movies_to_cluster, max_k=min(10, len(df_movies_to_cluster) -1))
    else:
        optimal_k_estimate = 2 # Default if not enough samples

    num_clusters = st.slider(
        "Select Number of Clusters (K):",
        min_value=2,
        max_value=8,
        value=2,
        step=1,
        key='num_clusters_slider_clustering_page'
    )

    # Perform clustering
    with st.spinner(f"Clustering movies into {num_clusters} groups... This might take a moment."):
        # Call perform_movie_clustering without anomaly removal parameters
        df_movies_clustered = clustering.perform_movie_clustering(
            df_movies_to_cluster, # This DF might already be winsorized
            n_clusters=num_clusters,
            remove_anomalies=False, # Explicitly set to False
            contamination_factor=0.0 # Irrelevant when remove_anomalies is False
        )

        # In this scenario, 'Cluster' will not be -1 due to Isolation Forest, only missing data
        df_clustered_successful = df_movies_clustered[df_movies_clustered['Cluster'] != -1].copy()
        df_unclustered_movies = df_movies_clustered[df_movies_clustered['Cluster'] == -1].copy() # Movies that couldn't be processed (e.g., no features)

        if not df_unclustered_movies.empty:
            st.warning(f"Note: {len(df_unclustered_movies)} movie(s) could not be clustered (e.g., due to entirely missing relevant features).")
            with st.expander(f"See {len(df_unclustered_movies)} Unclustered Movies"):
                st.dataframe(df_unclustered_movies[['Movie'] + box_office_cols + ['Genres']].sort_values(by='Worldwide_Adjusted', ascending=False))
        else:
            st.info("All movies in the selected range were successfully considered for clustering.")


        st.subheader("Cluster Overview: Average Metrics")
        if not df_clustered_successful.empty:
            cluster_metrics = df_clustered_successful.groupby('Cluster').agg(
                Avg_Worldwide=('Worldwide_Adjusted', 'mean'),
                Avg_Domestic=('Domestic_Adjusted', 'mean'),
                Avg_Foreign=('Foreign_Original', 'mean'), # Confirming Foreign_Original is used for display
                Movie_Count=('Movie', 'count')
            ).reset_index()

            for col in ['Avg_Worldwide', 'Avg_Domestic', 'Avg_Foreign']:
                cluster_metrics[col] = cluster_metrics[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(cluster_metrics.set_index('Cluster'))
        else:
            st.warning("No movies successfully clustered to display average metrics.")


        st.subheader("Cluster Overview: Top Genres")
        if not df_clustered_successful.empty:
            df_exploded_genres_clustered = df_clustered_successful.copy()
            # This logic correctly splits and prepares genres for counting,
            # assuming df_clustered_successful['Genres'] is already canonicalized.
            df_exploded_genres_clustered['Genres_List'] = df_exploded_genres_clustered['Genres'].astype(str).apply(
                lambda x: [g.strip() for g in x.split(',') if g.strip()] if pd.notna(x) and x != 'nan' else []
            )
            df_exploded_genres_clustered = df_exploded_genres_clustered.explode('Genres_List')

            top_genres_per_cluster = {}
            for cluster_id in sorted(df_clustered_successful['Cluster'].unique()):
                cluster_genres = df_exploded_genres_clustered[df_exploded_genres_clustered['Cluster'] == cluster_id][
                    'Genres_List']

                if not cluster_genres.empty:
                    genre_counts = cluster_genres.value_counts()
                    total_genre_occurrences = genre_counts.sum()

                    percentage_threshold = 0.05
                    min_genres_to_show = 2
                    max_genres_to_show = 4

                    dynamic_genres = []
                    for genre, count in genre_counts.items():
                        if (count / total_genre_occurrences >= percentage_threshold) and (len(dynamic_genres) < max_genres_to_show):
                            dynamic_genres.append(genre)
                        elif len(dynamic_genres) >= max_genres_to_show:
                            break
                        else:
                            break

                    if len(dynamic_genres) < min_genres_to_show and len(genre_counts) > 0:
                        dynamic_genres = genre_counts.nlargest(min_genres_to_show).index.tolist()

                    dynamic_genres = dynamic_genres[:max_genres_to_show]

                    if dynamic_genres:
                        top_genres_per_cluster[cluster_id] = ", ".join(dynamic_genres)
                    else:
                        top_genres_per_cluster[cluster_id] = "N/A"
                else:
                    top_genres_per_cluster[cluster_id] = "N/A"

            st.dataframe(
                pd.DataFrame.from_dict(top_genres_per_cluster, orient='index', columns=['Top Genres']).rename_axis(
                    'Cluster'))
        else:
            st.warning("No movies successfully clustered to display top genres.")

        st.subheader("Visualizing Clusters (Box Office)")
        if not df_clustered_successful.empty and 'Cluster' in df_clustered_successful.columns:
            fig_cluster_scatter = px.scatter(
                df_clustered_successful,
                x='Foreign_Original', # Confirming Foreign_Original is used for x-axis
                y='Domestic_Adjusted',
                color='Cluster',
                size='Worldwide_Adjusted',
                hover_name='Movie',
                hover_data={
                    'Worldwide_Adjusted': ':.2s',
                    'Domestic_Adjusted': ':.2s',
                    'Foreign_Original': ':.2s', # Confirming Foreign_Original for hover data
                    'Genres': True,
                    'Cluster': False
                },
                title=f"Movie Clusters by Box Office Performance and Genre ({num_clusters} Clusters)",
                labels={
                    'Worldwide_Adjusted': 'Worldwide Box Office (Adjusted)',
                    'Domestic_Adjusted': 'Domestic Box Office (Adjusted)',
                    'Foreign_Original': 'Foreign Box Office (Original)' # Changed label to reflect 'Original'
                },
                color_continuous_scale='picnic'
            )
            fig_cluster_scatter.update_layout(showlegend=True)
            st.plotly_chart(fig_cluster_scatter, use_container_width=True)
        else:
            st.warning("Not enough movies successfully clustered to create visualization.")

else:
    st.info("No movie data available for clustering in the selected year range.")

st.markdown("---")
st.caption("Data Source: Box Office Mojo, The World Bank, Kaggle | Analysis by Sebastian Abarca")