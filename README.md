# Smart Manufacturing Data Platform

End-to-end demo platform for manufacturing production analytics, built with Python, PostgreSQL, and Streamlit.

This project simulates engine production data, loads it into a star-schema data warehouse, runs anomaly and data-quality checks, and exposes an interactive analytics dashboard.

## Features

- **Synthetic data generator** – creates realistic engine production records (torque, temperature, cycle time, defects, downtime).
- **PostgreSQL star schema** – FactProduction + DimDate, DimMachine, DimOperator.
- **ETL pipeline** – pandas + SQLAlchemy for extract / transform / load.
- **Anomaly detection** – Z-score and Isolation Forest-based detection persisted to `ProductionAnomalies`.
- **Data quality checks** – missingness, defect rates, torque drift by machine exported as CSV.
- **Streamlit dashboard** – KPIs, trends, distributions, and downtime analytics over the warehouse.

## Architecture

```text
+-----------------+      +----------------------+      +------------------+
| Data Generator  | ---> |  raw_production_     | ---> |  ETL Pipeline    |
| (data_generator)|      |  data.csv (pandas)   |      |  (etl_pipeline)  |
+-----------------+      +----------------------+      +------------------+
                                                            |
                                                            v
                                                    +-------------------+
                                                    | PostgreSQL DW     |
                                                    |  DimDate          |
                                                    |  DimMachine       |
                                                    |  DimOperator      |
                                                    |  FactProduction   |
                                                    |  ProductionAnoms  |
                                                    +-------------------+
                                                          ^        ^
                                                          |        |
                                   +----------------------+        +----------------------+
                                   |                                             |
                           +-------------------+                     +-------------------+
                           | Data Quality      |                     | Streamlit         |
                           | Checks            |                     | Dashboard         |
                           | (data_quality)    |                     | (dashboard)       |
                           +-------------------+                     +-------------------+
```

## Repository Structure

```text
smart_data_project/
├── schema.sql                  # PostgreSQL star schema (Fact/Dim tables)
├── data_generator.py           # Generate synthetic production data (CSV)
├── etl_pipeline.py             # ETL: CSV -> PostgreSQL DW
├── anomaly_detection.py        # Z-score + Isolation Forest anomalies
├── data_quality.py             # Data quality checks & CSV report
├── dashboard.py                # Streamlit analytics dashboard
├── DETAIL_SETUP.md             # Detailed setup & run instructions
├── requirements.txt            # Python dependencies
├── .gitignore
└── .venv/ (local virtualenv, not tracked)
```

## Quickstart

### 1. Clone and create virtualenv

```bash
git clone https://github.com/singhvish1/Smart-Manufacturing-Data-Platform.git
cd Smart-Manufacturing-Data-Platform

# Recommended: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```bash
cd Smart-Manufacturing-Data-Platform/smart_data_project

# Recommended: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
   ```sql
   CREATE DATABASE manufacturing_dw;
   ```

2. Set connection variables in your shell (adjust as needed):

   ```bash
   export PGHOST=localhost
   export PGPORT=5432
   export PGUSER="$USER"
   unset PGPASSWORD
   export PGDATABASE=manufacturing_dw
   ```

3. Apply the warehouse schema (from the project root):

   ```bash
   psql "$PGDATABASE" -f schema.sql
   ```

### 4. Run the pipeline

From the project root (`smart_data_project`):

```bash
# Generate synthetic data
python data_generator.py

# Load into PostgreSQL DW
python etl_pipeline.py

# Run anomaly detection
python anomaly_detection.py

# Run data quality checks
python data_quality.py
```

### 5. Launch the dashboard

From the project root, with the virtualenv active:

```bash
streamlit run dashboard.py
```

This starts a local web app with KPIs, trends, defect analysis, and more. See `DETAIL_SETUP.md` for additional details and troubleshooting steps.
