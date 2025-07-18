import streamlit as st
import os

# --- Section to display dataPreparation.py code ---
st.markdown("---") # Optional separator
st.header("🔍 Data Preparation Script")
st.write("Below is the source code for the `dataPreparation.py` script used in this project:")

script_file_path = "dataPreparation.py" # Adjust this path if your file is in a subfolder (e.g., "scripts/dataPreparation.py")

if os.path.exists(script_file_path):
    try:
        with open(script_file_path, "r") as file:
            code_content = file.read()
        st.code(code_content, language="python")
    except Exception as e:
        st.error(f"Error reading dataPreparation.py: {e}")
else:
    st.warning(f"Could not find '{script_file_path}'. Please ensure the file exists at this path relative to your Streamlit app.")
