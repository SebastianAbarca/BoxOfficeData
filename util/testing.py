import pandas as pd
import re
import os


# --- Define the canonicalize_genre_string function ---
def canonicalize_genre_string(genres_str):
    if pd.isna(genres_str) or genres_str.strip() == '':
        return ''
    # Remove any non-alphanumeric characters (except comma and space), then split and strip
    # This also handles extra spaces, special characters that might interfere with sorting
    cleaned_str = re.sub(r'[^\w, ]', '', genres_str)
    cleaned_genres = [g.strip() for g in cleaned_str.split(',') if g.strip()]
    return ", ".join(sorted(cleaned_genres))


# --- Data Loading and Application of Canonicalization ---
def load_and_canonicalize_data(csv_path='../data/genre_bridge.csv'):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure 'data/final_box_office.csv' exists relative to where you run this script.")
        return None

    df_final = pd.read_csv(csv_path)

    # Apply the canonicalization function
    print(f"Applying genre canonicalization to '{csv_path}'...")
    df_final['Genres_Original'] = df_final['Genres']  # Keep original for comparison
    df_final['Genres'] = df_final['Genres'].apply(canonicalize_genre_string)
    print("Canonicalization applied.")

    return df_final


# --- Main Debugging Logic ---
if __name__ == "__main__":
    print("Starting genre canonicalization debugging script...")

    # Load data and apply canonicalization
    df_debug = load_and_canonicalize_data()

    if df_debug is None:
        exit()  # Exit if data loading failed

    print("\n--- Testing canonicalization on specific known permutations ---")
    test_cases = [
        "Action, Adventure",
        "Adventure, Action",
        " Science Fiction , Action ,Adventure ",
        "Adventure, Fantasy, Action",
        "Action, Fantasy, Adventure",
        "Comedy, Family, Animation",
        "Animation, Family, Comedy"
    ]

    for tc in test_cases:
        canonical = canonicalize_genre_string(tc)
        print(f"Input: '{tc}' -> Canonical: '{canonical}'")

    print("\n--- Inspecting samples from the DataFrame ---")
    print("Checking first 10 rows with original and canonicalized genres:")
    print(df_debug[['Movie', 'Genres_Original', 'Genres']].head(10).to_string())

    print("\n--- Searching for specific movies/genres that might show inconsistencies ---")
    # You can add specific movie titles from your CSV here
    movies_to_check = [
        "Pirates of the Caribbean: At World's End",  # Example movie that might have multiple genres
        "Avatar",
        "Jurassic World"
    ]

    found_movies = df_debug[df_debug['Movie'].isin(movies_to_check)]
    if not found_movies.empty:
        print(found_movies[['Movie', 'Genres_Original', 'Genres']].to_string())
    else:
        print(f"No specified test movies found in the dataset: {movies_to_check}")

    print("\n--- Finding examples of possible un-canonicalized genres if they exist ---")
    # This will identify rows where the original and canonicalized string are DIFFERENT
    # (after removing extra spaces)
    # The goal is that if original was "Adventure, Action", canonical is "Action, Adventure"
    # If the problem persists, it means even "Action, Adventure" vs "Adventure, Action"
    # might become different canonical strings, which is what we want to catch.

    # First, let's normalize the 'original' string a bit for comparison
    df_debug['Genres_Normalized_For_Comparison'] = df_debug['Genres_Original'].astype(str).apply(
        lambda x: ", ".join(sorted([g.strip() for g in re.sub(r'[^\w, ]', '', x).split(',') if g.strip()]))
    )

    # Now, compare the full canonicalized string to this normalized original
    inconsistent_genres_df = df_debug[
        (df_debug['Genres_Original'].notna()) &
        (df_debug['Genres_Original'].astype(str).str.strip() != '') &
        (df_debug['Genres_Normalized_For_Comparison'] != df_debug['Genres'])
        ]

    if not inconsistent_genres_df.empty:
        print(
            "\n--- WARNING: Found inconsistencies where canonicalization didn't produce expected output for sorted order ---")
        print("These movies might have problematic original genre strings (e.g., hidden characters):")
        print(inconsistent_genres_df[['Movie', 'Genres_Original', 'Genres']].head(20).to_string())
    else:
        print("\n--- No obvious inconsistencies found in genre canonicalization within the DataFrame. ---")
        print(
            "This suggests the function itself is working, and the issue might be display-related in Streamlit, or very rare edge cases not caught by these checks.")

    print("\nDebugging script finished.")