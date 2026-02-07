"""Data quality checks for the manufacturing data warehouse.

This script connects to the PostgreSQL warehouse and computes:
- Missing value percentages per column
- Defect rates by assembly line
- Average torque by machine
- Torque-drift flags by machine based on Z-score

It writes a long-form CSV report summarizing all checks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


LOGGER = logging.getLogger(__name__)


@dataclass
class DbConfig:
	"""Database configuration for PostgreSQL."""

	host: str = os.getenv("PGHOST", "localhost")
	port: int = int(os.getenv("PGPORT", "5432"))
	user: str = os.getenv("PGUSER", "postgres")
	password: str = os.getenv("PGPASSWORD", "postgres")
	database: str = os.getenv("PGDATABASE", "manufacturing_dw")

	def sqlalchemy_url(self) -> str:
		return (
			f"postgresql+psycopg2://{self.user}:{self.password}"
			f"@{self.host}:{self.port}/{self.database}"
		)


def configure_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
	)


def get_engine(config: DbConfig) -> Engine:
	"""Create a SQLAlchemy engine for PostgreSQL."""

	LOGGER.info("Creating SQLAlchemy engine for data quality checks")
	return create_engine(config.sqlalchemy_url(), future=True)


def extract_data(engine: Engine) -> pd.DataFrame:
	"""Extract production fact data joined with machine dimension."""

	LOGGER.info("Extracting production data for quality checks")

	query = text(
		"""
		SELECT
			fp.production_id,
			fp.engine_id,
			fp.machine_id,
			m.assembly_line,
			m.machine_description,
			fp.operator_id,
			fp.date_id,
			fp.torque_value,
			fp.temperature_celsius,
			fp.cycle_time_seconds,
			fp.defect_flag,
			fp.downtime_minutes
		FROM FactProduction fp
		JOIN DimMachine m ON fp.machine_id = m.machine_id
		"""
	)

	with engine.connect() as conn:
		df = pd.read_sql(query, conn)

	if df.empty:
		LOGGER.warning("No production data found for quality checks")
	else:
		LOGGER.info("Loaded %d production rows for quality checks", len(df))

	return df


def compute_missing_percentages(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute percentage of missing values per column."""

	if df.empty:
		return pd.DataFrame(columns=["column", "missing_pct"])

	missing_pct = df.isna().mean() * 100.0
	result = (
		missing_pct.reset_index()
		.rename(columns={"index": "column", 0: "missing_pct"})
		.sort_values("missing_pct", ascending=False)
	)
	return result


def compute_defect_rate_by_assembly_line(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute defect rate per assembly line."""

	if df.empty:
		return pd.DataFrame(columns=["assembly_line", "defect_rate_pct", "count"])

	grouped = df.groupby("assembly_line").agg(
		count=("production_id", "size"),
		defects=("defect_flag", lambda x: x.fillna(False).sum()),
	)

	grouped["defect_rate_pct"] = grouped["defects"] / grouped["count"] * 100.0
	grouped = grouped.reset_index()[["assembly_line", "defect_rate_pct", "count"]]
	return grouped.sort_values("defect_rate_pct", ascending=False)


def compute_avg_torque_by_machine(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute average torque per machine."""

	if df.empty:
		return pd.DataFrame(columns=["machine_id", "avg_torque", "count"])

	grouped = df.groupby("machine_id").agg(
		avg_torque=("torque_value", "mean"),
		count=("production_id", "size"),
	)
	grouped = grouped.reset_index()
	return grouped.sort_values("avg_torque", ascending=False)


def flag_torque_drift_machines(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
	"""Flag machines whose average torque deviates strongly from overall mean.

	Uses a Z-score of the machine-level average torque.
	"""

	if df.empty:
		return pd.DataFrame(columns=["machine_id", "avg_torque", "zscore", "is_drift"])

	machine_stats = df.groupby("machine_id")["torque_value"].mean().rename("avg_torque")
	mu = machine_stats.mean()
	sigma = machine_stats.std(ddof=0)

	if sigma == 0 or np.isnan(sigma):
		LOGGER.warning("Global torque std is zero; no drift flags will be set")
		result = machine_stats.to_frame().reset_index()
		result["zscore"] = 0.0
		result["is_drift"] = False
		return result

	zscores = (machine_stats - mu) / sigma
	is_drift = zscores.abs() > z_threshold

	result = machine_stats.to_frame().reset_index()
	result["zscore"] = zscores.values
	result["is_drift"] = is_drift.values

	LOGGER.info("Machines flagged for torque drift: %d", int(is_drift.sum()))
	return result.sort_values("zscore", key=lambda s: s.abs(), ascending=False)


def build_summary_report(
	missing_pct: pd.DataFrame,
	defect_rates: pd.DataFrame,
	avg_torque: pd.DataFrame,
	torque_drift: pd.DataFrame,
) -> pd.DataFrame:
	"""Build a long-form summary report from individual metrics.

	The report has columns: scope, entity, metric, value.
	"""

	records: List[Dict[str, object]] = []

	# Missing percentages
	for _, row in missing_pct.iterrows():
		records.append(
			{
				"scope": "column_missingness",
				"entity": str(row["column"]),
				"metric": "missing_pct",
				"value": float(row["missing_pct"]),
			}
		)

	# Defect rates
	for _, row in defect_rates.iterrows():
		records.append(
			{
				"scope": "assembly_line_defect_rate",
				"entity": f"line_{int(row['assembly_line'])}",
				"metric": "defect_rate_pct",
				"value": float(row["defect_rate_pct"]),
			}
		)
		records.append(
			{
				"scope": "assembly_line_volume",
				"entity": f"line_{int(row['assembly_line'])}",
				"metric": "record_count",
				"value": int(row["count"]),
			}
		)

	# Average torque per machine
	for _, row in avg_torque.iterrows():
		records.append(
			{
				"scope": "machine_avg_torque",
				"entity": f"machine_{int(row['machine_id'])}",
				"metric": "avg_torque",
				"value": float(row["avg_torque"]),
			}
		)
		records.append(
			{
				"scope": "machine_volume",
				"entity": f"machine_{int(row['machine_id'])}",
				"metric": "record_count",
				"value": int(row["count"]),
			}
		)

	# Torque drift flags
	for _, row in torque_drift.iterrows():
		records.append(
			{
				"scope": "machine_torque_drift",
				"entity": f"machine_{int(row['machine_id'])}",
				"metric": "zscore",
				"value": float(row["zscore"]),
			}
		)
		records.append(
			{
				"scope": "machine_torque_drift",
				"entity": f"machine_{int(row['machine_id'])}",
				"metric": "is_drift",
				"value": bool(row["is_drift"]),
			}
		)

	report = pd.DataFrame.from_records(records)
	return report


def save_report(report: pd.DataFrame, output_path: Path | None = None) -> Path:
	"""Save the quality report to CSV and return its path."""

	if output_path is None:
		output_path = Path(__file__).with_name("data_quality_report.csv")

	LOGGER.info("Writing data quality report to %s", output_path)
	report.to_csv(output_path, index=False)
	return output_path


def main() -> None:
	configure_logging()
	LOGGER.info("Starting data quality checks")

	config = DbConfig()
	engine = get_engine(config)

	df = extract_data(engine)
	missing_pct = compute_missing_percentages(df)
	defect_rates = compute_defect_rate_by_assembly_line(df)
	avg_torque = compute_avg_torque_by_machine(df)
	torque_drift = flag_torque_drift_machines(df)

	report = build_summary_report(missing_pct, defect_rates, avg_torque, torque_drift)
	save_report(report)

	LOGGER.info("Data quality checks completed")


if __name__ == "__main__":
	main()
