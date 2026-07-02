import streamlit as st
import pandas as pd
import subprocess
import os
import time

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("🏆 The Singularity - Candidate Ranker")
st.markdown("Redrob Hackathon - Intelligent Candidate Discovery & Ranking Challenge")

st.sidebar.header("Controls")

candidates_file = "candidates.jsonl"
submission_file = "submission.csv"

# Button to run the ranker
if st.sidebar.button("🚀 Run Ranker"):
    if not os.path.exists(candidates_file):
        st.sidebar.error(f"Could not find {candidates_file}. Please ensure the data file is present.")
    else:
        st.sidebar.info("Running `rank.py`... This takes about ~30-40 seconds.")
        
        # Run the ranker as a subprocess
        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", "rank.py", "--candidates", candidates_file, "--out", submission_file],
                capture_output=True, text=True, check=True
            )
            elapsed = time.time() - start_time
            st.sidebar.success(f"Ranking completed in {elapsed:.1f} seconds!")
            with st.expander("Show Console Output"):
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.sidebar.error("An error occurred while running the ranker.")
            with st.expander("Show Error Details"):
                st.code(e.stderr)

st.divider()

# Display the results
st.header("Ranked Candidates (Top 100)")
if os.path.exists(submission_file):
    try:
        df = pd.read_csv(submission_file)
        
        # Display some quick metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates Ranked", len(df))
        col2.metric("Top Score", f"{df['score'].max():.4f}")
        col3.metric("Rank 100 Score", f"{df['score'].min():.4f}")
        
        st.markdown("### Submission Data")
        
        # Create a nice interactive dataframe view
        st.dataframe(
            df,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", help="Candidate Rank (1-100)"),
                "candidate_id": st.column_config.TextColumn("ID", help="Unique Candidate ID"),
                "score": st.column_config.NumberColumn("Score", format="%.5f"),
                "reasoning": st.column_config.TextColumn("Reasoning", help="Fact-grounded explanation")
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        st.download_button(
            label="📥 Download submission.csv",
            data=open(submission_file, "rb"),
            file_name="submission.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error reading {submission_file}: {e}")
else:
    st.info(f"No `{submission_file}` found. Click **Run Ranker** in the sidebar to generate the rankings.")
