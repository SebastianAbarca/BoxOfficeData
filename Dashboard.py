import streamlit as st
import pandas as pd
import plotly.express as px
from util import graphs, clustering


st.markdown("""
            <style>
            .stApp{
                background-color: #000926; 
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
            /* Styling for st.container elements */
            .stContainer {
                background-color: #05007D; /* A slightly lighter blue/purple for containers to stand out from background */
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* More pronounced shadow on dark background */
            }
            /* Specific border for the Title container (corrected color) */
            div[data-testid="stVerticalBlock"] > div:nth-child(1) .stContainer {
                border: 2px solid #FFFFFF; /* White border for the title container */
            }
            /* Further general text overrides if needed (e.g. for selectbox/slider labels) */
            .stSelectbox label {
                color: #FFFFFF;
            }
            .stSlider label {
                color: #FFFFFF;
            }
            </style>
            """
,unsafe_allow_html=True)
# --- Helper Function for YoY Delta Calculation ---
def calculate_yoy_delta(current_year_data_row, prev_year_data_df, yoy_column_name):
    delta_val = None
    delta_color = "off" # Default to 'off' if no valid delta can be calculated

    if not prev_year_data_df.empty:
        prev_year_value = prev_year_data_df.iloc[0][yoy_column_name]
        current_year_value = current_year_data_row[yoy_column_name]

        if pd.notna(current_year_value) and pd.notna(prev_year_value):
            delta_val = current_year_value - prev_year_value
            # For growth metrics, 'normal' behavior is desired:
            # Positive delta (increase) = Green
            # Negative delta (decrease) = Red
            delta_color = "normal"

    return delta_val, delta_color

# --- Load Data ---
@st.cache_data # Cache data loading for performance
def load_data():
    # Make sure your data files are in a 'data' subfolder, or adjust paths
    df_final = pd.read_csv('data/final_box_office.csv')
    df_yearly_totals = pd.read_csv('data/yearly_top10_summary.csv')
    df_genre_bridge = pd.read_csv('data/genre_bridge.csv')
    df_production_bridge = pd.read_csv('data/production_bridge.csv')

    # Ensure Year is int for filtering
    df_final['Year'] = df_final['Year'].astype(int)
    df_yearly_totals['Year'] = df_yearly_totals['Year'].astype(int)
    df_genre_bridge['Year'] = df_genre_bridge['Year'].astype(int)
    df_production_bridge['Year'] = df_production_bridge['Year'].astype(int) # Uncomment if Year column exists and needed

    # Sort yearly totals for correct latest year picking
    df_yearly_totals = df_yearly_totals.sort_values('Year').reset_index(drop=True)

    return df_final, df_yearly_totals, df_genre_bridge, df_production_bridge

df_final, df_yearly_totals, df_genre_bridge, df_production_bridge = load_data()

# --- Configuration ---
st.set_page_config(
    page_title="Film Box Office Analysis",
    layout="wide", # Use 'wide' for more horizontal space
    initial_sidebar_state="collapsed"
)

# --- Dashboard Layout Containers ---
Title_container = st.container()
Total_BO_container = st.container()
YoY_Growth_container = st.container()
Charts_container = st.container()
# --- Filters (Chart filters remain in sidebar) ---
latest_year = df_yearly_totals['Year'].max()
all_years = df_yearly_totals['Year'].unique()

# --- Dashboard Content ---
with Title_container:
    st.title("🎬 25-Year Global Box Office Insights")
    st.markdown("---") # Visual separator
with Total_BO_container:
    st.header("Total Box Office (All-Time)") # More descriptive header
    WwCol, DomCol, ForCol = st.columns(3)
    with WwCol:
        st.metric(
            label="Worldwide (Nominal)",
            value=f"${df_yearly_totals['Worldwide_Original'].sum():,.0f}"
        )
        st.metric(
            label="Worldwide (Real)",
            value=f"${df_yearly_totals['Worldwide_Adjusted'].sum():,.0f}"
        )
    with DomCol:
        st.metric(
            label="Domestic (Nominal)",
            value=f"${df_yearly_totals['Domestic_Original'].sum():,.0f}"
        )
        st.metric(
            label="Domestic (Real)",
            value=f"${df_yearly_totals['Domestic_Adjusted'].sum():,.0f}"
        )
    with ForCol:
        st.metric(
            label="Foreign (Nominal)",
            value=f"${df_yearly_totals['Foreign_Original'].sum():,.0f}"
        )
        # Adding a placeholder to balance the 2x3 grid
        st.metric(label="", value="") # Empty label and value
with YoY_Growth_container:
    selected_year_for_kpis = st.selectbox(
        "Select Year for YoY Growth Metrics:",
        options=all_years,
        index=len(all_years) - 1,
        key='kpi_year_selector'
    )
    col_kpi_year_filter, _ = st.columns([0.3, 0.7])
    kpi_data = df_yearly_totals[df_yearly_totals['Year'] == selected_year_for_kpis].iloc[0]
    prev_year_data_df_kpi = df_yearly_totals[df_yearly_totals['Year'] == (selected_year_for_kpis - 1)]

    st.subheader(f"Growth Metrics for {selected_year_for_kpis}")
    col_yoy_1, col_yoy_2, col_yoy_3, col_yoy_4, col_yoy_5 = st.columns(5)

    with col_yoy_1:
        delta_val, delta_color = calculate_yoy_delta(kpi_data, prev_year_data_df_kpi, 'YoY_Growth_Worldwide_Original')
        st.metric(
            label="Worldwide (Nominal)",
            value=f"{kpi_data['YoY_Growth_Worldwide_Original']:.1f}%" if pd.notna(
                kpi_data['YoY_Growth_Worldwide_Original']) else "N/A",
            delta=f"{delta_val:.1f}% (vs. prev YoY)" if delta_val is not None else None,
            delta_color=delta_color
        )
    with col_yoy_2:
        delta_val, delta_color = calculate_yoy_delta(kpi_data, prev_year_data_df_kpi, 'YoY_Growth_Domestic_Original')
        st.metric(
            label="Domestic (Nominal)",
            value=f"{kpi_data['YoY_Growth_Domestic_Original']:.1f}%" if pd.notna(
                kpi_data['YoY_Growth_Domestic_Original']) else "N/A",
            delta=f"{delta_val:.1f}% (vs. prev YoY)" if delta_val is not None else None,
            delta_color=delta_color
        )
    with col_yoy_3:
        delta_val, delta_color = calculate_yoy_delta(kpi_data, prev_year_data_df_kpi, 'YoY_Growth_Foreign')
        st.metric(
            label="Foreign (Nominal)",
            value=f"{kpi_data['YoY_Growth_Foreign']:.1f}%" if pd.notna(kpi_data['YoY_Growth_Foreign']) else "N/A",
            delta=f"{delta_val:.1f}% (vs. prev YoY)" if delta_val is not None else None,
            delta_color=delta_color
        )
    with col_yoy_4:
        delta_val, delta_color = calculate_yoy_delta(kpi_data, prev_year_data_df_kpi, 'YoY_Growth_Worldwide_Adjusted')
        st.metric(
            label="Worldwide (Real)",
            value=f"{kpi_data['YoY_Growth_Worldwide_Adjusted']:.1f}%" if pd.notna(
                kpi_data['YoY_Growth_Worldwide_Adjusted']) else "N/A",
            delta=f"{delta_val:.1f}% (vs. prev YoY)" if delta_val is not None else None,
            delta_color=delta_color
        )
    with col_yoy_5:
        delta_val, delta_color = calculate_yoy_delta(kpi_data, prev_year_data_df_kpi, 'YoY_Growth_Domestic_Adjusted')
        st.metric(
            label="Domestic (Real)",
            value=f"{kpi_data['YoY_Growth_Domestic_Adjusted']:.1f}%" if pd.notna(
                kpi_data['YoY_Growth_Domestic_Adjusted']) else "N/A",
            delta=f"{delta_val:.1f}% (vs. prev YoY)" if delta_val is not None else None,
            delta_color=delta_color
        )
with Charts_container:
    st.header("Key Trends & Insights")

    year_range = st.slider(
        "Select Year Range for Charts:",
        min_value=int(df_yearly_totals['Year'].min()),
        max_value=int(df_yearly_totals['Year'].max()),
        value=(int(df_yearly_totals['Year'].min()), int(df_yearly_totals['Year'].max()))
    )

    df_yearly_filtered = df_yearly_totals[
        (df_yearly_totals['Year'] >= year_range[0]) & (df_yearly_totals['Year'] <= year_range[1])]
    df_final_filtered_charts = df_final[(df_final['Year'] >= year_range[0]) & (df_final['Year'] <= year_range[1])]

    # --- Chart 1: Top 10 movies year_range ---
    st.subheader(f"Top 10 Movies ({year_range[0]} to {year_range[1]})")
    df_recent_movies = df_final_filtered_charts.copy()

    df_recent_movies_agg = df_recent_movies.groupby('Movie').agg(
        Worldwide_Total=('Worldwide_Corrected', 'sum')
    ).reset_index()

    df_top_10_movies = df_recent_movies_agg.nlargest(10, 'Worldwide_Total')

    fig_top_movies = px.bar(
        df_top_10_movies,
        x='Worldwide_Total',
        y='Movie',
        orientation='h',
        title=f"Top 10 Movies by Worldwide Box Office ({year_range[0]}-{year_range[1]})",
        labels={'Worldwide_Total': 'Worldwide Box Office ($)', 'Movie': 'Movie Title'},
        color='Worldwide_Total', # Color by value
        color_continuous_scale=px.colors.sequential.Inferno # Use a nice color scale
    )
    fig_top_movies.update_layout(showlegend=False) # No need for color legend
    fig_top_movies.update_yaxes(autorange="reversed") # Puts highest at top
    st.plotly_chart(fig_top_movies, use_container_width=True)

    st.markdown("---")

    # --- Chart 2: Genre Market Share last 5 years ---
    st.subheader("Genre Market Share")

    five_years_ago_chart = max(df_final_filtered_charts['Year'].min(), df_final_filtered_charts['Year'].max() - 4)
    df_genre_filtered = df_final_filtered_charts[(df_final_filtered_charts['Year'] >= five_years_ago_chart)].copy()

    df_genre_filtered['Genres'] = df_genre_filtered['Genres'].astype(str)
    df_exploded_genres = df_genre_filtered.assign(Genres=df_genre_filtered['Genres'].str.split(',')).explode('Genres')
    df_exploded_genres['Genres'] = df_exploded_genres['Genres'].str.strip()


    genre_yearly_totals_chart = df_exploded_genres.groupby(['Year', 'Genres']).agg(
        Total_Worldwide=('Worldwide_Original', 'sum')
    ).reset_index()

    total_worldwide_per_year_chart = genre_yearly_totals_chart.groupby('Year')['Total_Worldwide'].sum().reset_index()
    total_worldwide_per_year_chart.rename(columns={'Total_Worldwide': 'Overall_Worldwide'}, inplace=True)

    genre_market_share = genre_yearly_totals_chart.merge(total_worldwide_per_year_chart, on='Year')
    genre_market_share['Market_Share'] = (genre_market_share['Total_Worldwide'] / genre_market_share['Overall_Worldwide']) * 100

    # To prevent too many genres, focus on top N or filter smaller ones
    top_genres = genre_market_share.groupby('Genres')['Market_Share'].sum().nlargest(10).index # Adjust N if too many lines
    genre_market_share_filtered = genre_market_share[genre_market_share['Genres'].isin(top_genres)]

    fig_genre_share = px.area(
        genre_market_share_filtered,
        x='Year',
        y='Market_Share',
        color='Genres',
        title=f"Worldwide Box Office Market Share by Genre ({five_years_ago_chart}-{df_final_filtered_charts['Year'].max()})",
        labels={'Market_Share': 'Market Share (%)', 'Year': 'Year'},
        line_group='Genres',
        hover_name='Genres'
    )
    fig_genre_share.update_layout(hovermode="x unified")
    fig_genre_share.update_yaxes(range=[0,100], ticksuffix="%")
    fig_genre_share.update_xaxes(dtick=1)
    st.plotly_chart(fig_genre_share, use_container_width=True)

    st.subheader("Genre/Box Office Heatmap :fire:")
    freq_matrix, avg_bo_matrix, mask = graphs.compute_genre_cooccurrence_bo(df_final_filtered_charts)
    fig_heatmap = graphs.plot_heatmap_with_annotations(avg_bo_matrix, freq_matrix, mask)
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.caption("Data Source: Box Office Mojo, The World Bank, Kaggle | Dashboard by Sebastian Abarca")