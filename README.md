# Modular Object-Oriented Data Science Pipeline

A production-grade, reusable, and tested data pipeline built in Python using Scikit-Learn and Pandas. This project follows object-oriented programming (OOP) principles to prevent data leakage, handle missing values, drop duplicates, scale numerical features, and encode categorical variables cleanly between training and inference phases.

---

## Project Structure

```text
my_ds_project/
│
├── data/
│   ├── raw/                 # Original, immutable data inputs
│   └── processed/           # Cleaned and feature-engineered outputs
│
├── src/
│   ├── __init__.py
│   ├── loader.py            # DataLoader class for data ingestion
│   ├── cleaner.py           # DataCleaner class for missing values & deduplication
│   └── transformer.py       # FeatureTransformer class for scaling & encoding
│
├── tests/
│   └── test_components.py   # Unit tests for individual pipeline modules
│
├── main.py                  # End-to-end pipeline orchestrator
├── test_pipeline.py         # End-to-end test and validation script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

---

## Getting Started

### 1. Prerequisites

Ensure you have **Python 3.10+** installed on your machine.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt

```

### 3. Running the Pipeline

Execute the main orchestrator script to load, clean, transform, and export data:

```bash
python main.py

```

### 4. Running Unit Tests

Validate all modules independently using `pytest`:

```bash
pytest

```

---

## Architectural Highlights

* **Scikit-Learn Compatibility:** Custom components inherit from `BaseEstimator` and `TransformerMixin`, making them fully compatible with machine learning estimators and production pipelines.
* **Data Leakage Prevention:** Transformations are cleanly fitted on training subsets and applied independently during inference.
* **Robust Testing:** Fully covered by automated `pytest` unit tests ensuring zero missing values and correct data shapes post-processing.
