import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest  # NEW: Import IsolationForest
import numpy as np
import warnings

# Suppress warnings that might clutter Streamlit output, e.g., from n_init for KMeans
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# --- NEW FUNCTION: Winsorization ---
def winsorize_box_office(df: pd.DataFrame, columns: list, limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    df_winsorized = df.copy()
    lower_bound_p, upper_bound_p = limits

    for col in columns:
        if col in df_winsorized.columns and pd.api.types.is_numeric_dtype(df_winsorized[col]):
            # Calculate bounds
            lower_val = df_winsorized[col].quantile(lower_bound_p)
            upper_val = df_winsorized[col].quantile(upper_bound_p)

            # Apply capping
            df_winsorized[col] = np.clip(df_winsorized[col], lower_val, upper_val)
        else:
            print(f"Warning: Column '{col}' not found or not numeric, skipping winsorization for it.")
    return df_winsorized


# --- MODIFIED FUNCTION: perform_movie_clustering to include Isolation Forest ---
def perform_movie_clustering(
        df_movies: pd.DataFrame,
        n_clusters: int = 5,
        remove_anomalies: bool = False,  # NEW: Control for Isolation Forest
        contamination_factor: float = 0.01  # NEW: Contamination parameter for Isolation Forest
) -> pd.DataFrame:
    df_cluster = df_movies.copy()

    # Define the numerical and categorical columns to be used for clustering
    # IMPORTANT: Use 'Foreign_Adjusted' for consistency if available, otherwise 'Foreign_Original'
    numerical_cols = ['Worldwide_Adjusted', 'Domestic_Adjusted', 'Foreign_Adjusted']  # Changed to Adjusted
    categorical_cols = ['Genres']

    # --- 1. Feature Selection and Cleaning ---
    # Handle missing numerical values: Impute with median for robustness
    for col in numerical_cols:
        if col in df_cluster.columns:
            # Ensure the column is numeric; if not, coerce errors
            df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
            df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())
        else:
            print(f"Warning: Numerical column '{col}' not found in DataFrame. Adding a dummy column with 0s.")
            df_cluster[col] = 0.0  # Add as float to match other numerical data

    # Handle missing Genres: Convert to empty list if NaN for MultiLabelBinarizer
    if 'Genres' in df_cluster.columns:
        # Canonicalization should ideally happen much earlier (in load_data),
        # but defensively ensure it's a list of strings for MLB.
        df_cluster['Genres_List'] = df_cluster['Genres'].astype(str).apply(
            lambda x: [g.strip() for g in x.split(',') if g.strip()] if pd.notna(x) and x != 'nan' else []
        )
    else:
        print("Warning: 'Genres' column not found in DataFrame. Clustering will proceed without genre features.")
        df_cluster['Genres_List'] = [[] for _ in range(len(df_cluster))]

    # Filter out movies that have absolutely no valid features to cluster on
    # (e.g., all numericals are NaN/0 and no genres)
    initial_valid_rows_mask = (
            (df_cluster[numerical_cols].sum(axis=1) != 0) |
            (df_cluster['Genres_List'].apply(len) > 0)
    )
    df_processed_for_clustering = df_cluster[initial_valid_rows_mask].copy()

    if df_processed_for_clustering.empty:
        print("Warning: No movies left after initial data cleaning. Returning original DataFrame with all Cluster -1.")
        df_movies['Cluster'] = -1
        return df_movies

    # --- 2. Feature Engineering / Data Transformation ---

    # a. Numerical Feature Scaling
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(df_processed_for_clustering[numerical_cols])
    df_scaled_numerical = pd.DataFrame(scaled_numerical_features, columns=numerical_cols,
                                       index=df_processed_for_clustering.index)

    # b. Categorical Feature Encoding (Multi-Label Binarizer for Genres)
    mlb_genres = MultiLabelBinarizer()
    genre_features = mlb_genres.fit_transform(df_processed_for_clustering['Genres_List'])
    df_genre_encoded = pd.DataFrame(genre_features, columns=mlb_genres.classes_,
                                    index=df_processed_for_clustering.index)

    # --- c. Feature Concatenation ---
    X_combined = pd.concat([df_scaled_numerical, df_genre_encoded], axis=1)

    # Remove any columns that are all zeros after encoding (e.g., genres that never appeared in the filtered data)
    X_combined = X_combined.loc[:, (X_combined != 0).any(axis=0)]

    # Handle case where X_combined might become empty or have too few features
    if X_combined.empty or X_combined.shape[1] == 0:
        print("Warning: No valid features to cluster on after feature engineering. Assigning Cluster -1 to all movies.")
        df_movies['Cluster'] = -1
        return df_movies

    # --- 3. Dimensionality Reduction (PCA) ---
    # Max components can't exceed min(n_samples-1, n_features-1, max_cap)
    n_components = min(X_combined.shape[0] - 1, X_combined.shape[1] - 1, 50)
    if n_components <= 0:
        print(
            f"Not enough data/features for PCA (samples: {X_combined.shape[0]}, features: {X_combined.shape[1]}). Skipping PCA.")
        X_final_features = X_combined
    else:
        pca = PCA(n_components=n_components, random_state=42)
        X_final_features = pca.fit_transform(X_combined)

    # --- 4. Anomaly Detection (Isolation Forest) ---
    # Perform anomaly detection on the PCA-transformed features
    if remove_anomalies and X_final_features.shape[0] > 1:  # Need at least 2 samples for IsolationForest
        try:
            iforest = IsolationForest(contamination=contamination_factor, random_state=42, n_jobs=-1)
            # -1 indicates an outlier, 1 indicates an inlier
            outlier_predictions = iforest.fit_predict(X_final_features)

            # Filter the data to keep only inliers for clustering
            df_inliers_for_clustering = df_processed_for_clustering[outlier_predictions == 1].copy()
            X_final_features_inliers = X_final_features[outlier_predictions == 1]

            if df_inliers_for_clustering.empty or X_final_features_inliers.shape[0] < n_clusters:
                print(
                    f"Warning: After anomaly detection, only {X_final_features_inliers.shape[0]} inliers remain, which is less than n_clusters ({n_clusters}). Cannot perform K-Means. All movies will be assigned Cluster -1.")
                df_movies['Cluster'] = -1
                return df_movies

            # These are the movies that will be successfully clustered
            df_clustered_temp = df_inliers_for_clustering
            X_data_for_kmeans = X_final_features_inliers

        except Exception as e:
            print(f"Error during Isolation Forest anomaly detection: {e}. Skipping anomaly removal.")
            df_clustered_temp = df_processed_for_clustering.copy()
            X_data_for_kmeans = X_final_features
    else:
        df_clustered_temp = df_processed_for_clustering.copy()
        X_data_for_kmeans = X_final_features

    # Ensure enough samples for K-Means after all preprocessing
    if X_data_for_kmeans.shape[0] < n_clusters or X_data_for_kmeans.shape[0] < 2:
        print(
            f"Warning: Not enough valid samples ({X_data_for_kmeans.shape[0]}) for K-Means with {n_clusters} clusters. Assigning Cluster -1 to all movies.")
        df_movies['Cluster'] = -1
        return df_movies

    # --- 5. Clustering Algorithm Application (K-Means) ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered_temp['Cluster'] = kmeans.fit_predict(X_data_for_kmeans)

    # --- 6. Merge Clusters back to original DataFrame ---
    # Initialize all original movies with -1 (unclustered/anomaly)
    df_movies_with_clusters = df_movies.copy()
    df_movies_with_clusters['Cluster'] = -1

    # Update cluster labels for movies that were successfully processed and clustered
    # Use df_clustered_temp's index to match back to the original df_movies
    df_movies_with_clusters.loc[df_clustered_temp.index, 'Cluster'] = df_clustered_temp['Cluster']

    return df_movies_with_clusters


# --- MODIFIED FUNCTION: get_best_k ---
def get_best_k(df_movies: pd.DataFrame, max_k: int = 10) -> int:
    df_cluster_temp = df_movies.copy()

    # IMPORTANT: Use 'Foreign_Adjusted' for consistency
    numerical_cols = ['Worldwide_Adjusted', 'Domestic_Adjusted', 'Foreign_Adjusted']  # Changed to Adjusted

    # Handle missing numerical values: Impute with median for robustness
    for col in numerical_cols:
        if col in df_cluster_temp.columns:
            df_cluster_temp[col] = pd.to_numeric(df_cluster_temp[col], errors='coerce')
            df_cluster_temp[col] = df_cluster_temp[col].fillna(df_cluster_temp[col].median())
        else:
            print(f"Warning: Numerical column '{col}' not found for get_best_k. Adding a dummy column.")
            df_cluster_temp[col] = 0.0

    if 'Genres' in df_cluster_temp.columns:
        df_cluster_temp['Genres_List'] = df_cluster_temp['Genres'].astype(str).apply(
            lambda x: [g.strip() for g in x.split(',') if g.strip()] if pd.notna(x) and x != 'nan' else []
        )
    else:
        df_cluster_temp['Genres_List'] = [[] for _ in range(len(df_cluster_temp))]

    # Filter out movies that have absolutely no valid features to cluster on
    initial_valid_rows_mask = (
            (df_cluster_temp[numerical_cols].sum(axis=1) != 0) |
            (df_cluster_temp['Genres_List'].apply(len) > 0)
    )
    df_processed_for_k = df_cluster_temp[initial_valid_rows_mask].copy()

    if df_processed_for_k.empty:
        return 2  # No data to meaningfully cluster, return default k

    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(df_processed_for_k[numerical_cols])
    df_scaled_numerical = pd.DataFrame(scaled_numerical_features, columns=numerical_cols,
                                       index=df_processed_for_k.index)

    mlb_genres = MultiLabelBinarizer()
    genre_features = mlb_genres.fit_transform(df_processed_for_k['Genres_List'])
    df_genre_encoded = pd.DataFrame(genre_features, columns=mlb_genres.classes_, index=df_processed_for_k.index)

    X_combined = pd.concat([df_scaled_numerical, df_genre_encoded], axis=1)
    X_combined = X_combined.loc[:, (X_combined != 0).any(axis=0)]  # Remove all-zero columns

    if X_combined.empty or X_combined.shape[1] == 0:
        print("Warning: No valid features to cluster on for get_best_k. Returning default K=2.")
        return 2

    n_components = min(X_combined.shape[0] - 1, X_combined.shape[1] - 1, 50)
    if n_components <= 0:
        print(
            f"Not enough data/features for PCA in get_best_k (samples: {X_combined.shape[0]}, features: {X_combined.shape[1]}). Skipping PCA.")
        X_processed = X_combined
    else:
        pca = PCA(n_components=n_components, random_state=42)
        X_processed = pca.fit_transform(X_combined)

    if X_processed.shape[0] < 2:
        return 2  # Need at least 2 samples for silhouette score and clustering

    silhouette_scores = []
    # Ensure k_range is valid: starts from 2, and does not exceed number of samples - 1
    k_range = range(2, min(max_k + 1, X_processed.shape[0]))
    if len(k_range) < 1:  # If range is empty, means max_k is too small for data size
        return 2

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_processed)
            # Silhouette score requires at least 2 unique labels, and more than 1 sample per label
            # Check if all clusters have at least one point, and if there's more than one cluster formed
            if len(np.unique(clusters)) > 1 and np.all(np.bincount(clusters) > 0):
                score = silhouette_score(X_processed, clusters)
                silhouette_scores.append((k, score))
            else:
                silhouette_scores.append((k, -1.0))  # Invalid score if not enough unique clusters/points
        except Exception as e:
            print(f"Error calculating silhouette for k={k}: {e}")
            silhouette_scores.append((k, -1.0))  # Assign a low score if error occurs

    if any(s[1] > -1 for s in silhouette_scores):  # Check if any valid scores were computed
        # Filter out invalid scores before finding the maximum
        valid_scores = [s for s in silhouette_scores if s[1] > -1]
        best_k = max(valid_scores, key=lambda item: item[1])[0]
        return best_k
    return 2  # Default if no valid scores could be calculated or if list is empty