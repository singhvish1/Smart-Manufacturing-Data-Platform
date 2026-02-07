# Smart Data Project – Manufacturing Pipeline

This directory contains a small end-to-end manufacturing analytics stack:

- Synthetic data generator
- PostgreSQL star-schema warehouse
- ETL pipeline
- Anomaly detection
- Data quality checks
- Streamlit dashboard

## 1. Prerequisites

- Python 3.9+
- PostgreSQL (local or remote)
- `pip` for installing Python packages

Install Python dependencies (from this `data` directory):

```bash
pip install -r requirements.txt
```

## 2. PostgreSQL Setup

1. Create a database (default name used by the code is `manufacturing_dw`):

	```sql
	CREATE DATABASE manufacturing_dw;
	```

2. Set environment variables (optional – otherwise defaults are used):

	```bash
	export PGHOST=localhost
	export PGPORT=5432
	export PGUSER=postgres
	export PGPASSWORD=postgres
	export PGDATABASE=manufacturing_dw
	```

3. Apply the warehouse schema:

	```bash
	psql "$PGDATABASE" -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -f schema.sql
	```

Make sure the PostgreSQL user has permission to create tables and insert data.

## 3. Data Generation

From this `data` directory, generate synthetic production data:

```bash
python data_generator.py
```

This creates `raw_production_data.csv` with 10,000 engine production records.

## 4. ETL Pipeline

Load the raw CSV into the PostgreSQL star schema:

```bash
python etl_pipeline.py
```

This populates the `DimDate`, `DimMachine`, `DimOperator`, and `FactProduction` tables.

## 5. Anomaly Detection

Run anomaly detection (Z‑score and Isolation Forest) and persist anomalies:

```bash
python anomaly_detection.py
```

This writes anomalies into the `ProductionAnomalies` table and saves
`anomaly_counts_over_time.png` in this directory.

## 6. Data Quality Checks

Compute data quality metrics and export a summary CSV:

```bash
python data_quality.py
```

This produces `data_quality_report.csv` with missing percentages, defect rates,
average torque by machine, and torque drift flags.

## 7. Streamlit Dashboard

Start the interactive dashboard:

```bash
streamlit run dashboard.py
```

The dashboard connects to the same PostgreSQL database and provides:

- KPI cards (total engines, defect rate, average cycle time, total downtime)
- Torque trend line chart
- Defect rate by assembly line
- Cycle time distribution
- Top 5 machines by total downtime

Use the sidebar filters to narrow by date range and assembly line.
