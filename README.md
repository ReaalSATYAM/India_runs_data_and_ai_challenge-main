# Redrob Hackathon - Intelligent Candidate Discovery & Ranking

**Team Name:** The Singularity  
**Team Leader:** Satyam Naithani  

## Overview
This repository contains a deterministic, feature-based candidate ranker designed to evaluate 100,000 resume profiles against a Senior AI Engineer Job Description. 

Our approach prioritizes **interpretability and adversarial robustness**. Instead of using black-box LLMs or embedding similarities (which are easily defeated by keyword stuffers and fake honeypot profiles), we use a structured scoring system. Every ranking decision is auditable, explainable, and fact-grounded.

## Features
- **Anti-Stuffer Firewall**: Career evidence and job title history act as a multiplicative gate, effectively zeroing out non-tech profiles that try to game the system with AI buzzwords.
- **Honeypot Immunity**: A pre-screening `consistency_gate` evaluates impossible timelines (e.g., 5 years of experience but using a tool for 8 years). Found 48 honeypots with zero false positives.
- **Zero-Hallucination Reasoning**: Explanations are built entirely by substring matching against the candidate's actual profile fields.
- **High Performance**: Pure Python streaming architecture. Requires <1GB RAM and processes 100K profiles in ~32 seconds on a single CPU core.

## Usage

### 1. Command-Line Ranker (Headless)
The core logic resides in `rank.py`. You can run the ranking algorithm directly from your terminal. It will stream the dataset and output a ranked CSV.

```bash
# Ensure you have the dataset available in the root folder
python rank.py --candidates candidates.jsonl --out submission.csv
```

### 2. Streamlit Interactive App
We have provided a simple web UI using Streamlit to interactively trigger the ranking and view the results in a clean table format.

**Prerequisites:**
```bash
pip install streamlit pandas
```

**Run the App:**
```bash
streamlit run app.py
```
This will open a local web server (typically `http://localhost:8501`) where you can click a button to run the ranker and visualize the top 100 candidates alongside their generated reasoning.
**Note:** Using the streamlit app will take more time to compute(80~90 sec), so using the command line is recomended for fast inferrence.   
## Audit & Evaluation Scripts
- `profile_data.py`: Performs EDA on the dataset to understand distributions.
- `audit_honeypots_full_pool.py`: Scans all 100K profiles and dumps identified honeypots to a CSV for manual inspection.
- `check_honeypots.py`: Validates that the final `submission.csv` contains a 0% honeypot rate.
- `validate_submission.py`: Checks the final output against hackathon formatting constraints.
