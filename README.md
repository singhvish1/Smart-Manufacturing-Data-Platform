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

For full environment setup and run instructions, see `DETAIL_SETUP.md`.
