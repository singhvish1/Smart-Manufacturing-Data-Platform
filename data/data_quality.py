"""Data quality checks for the manufacturing production warehouse.

This script connects to the PostgreSQL data warehouse, computes several
data quality metrics, and writes a long-form summary report as a CSV.

Checks performed
----------------
- Percentage of missing values per column
- Defect rate per assembly line
- Average torque per machine
- Machines with abnormal torque drift (Z-score-based)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


LOGGER = logging.getLogger(__name__)


@dataclass
class DbConfig:
	"""Database connection configuration for PostgreSQL."""

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
	"""Extract production data with machine information from the warehouse."""

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

	LOGGER.info("Loaded %d production rows", len(df))
	return df


def compute_missing_percentages(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute percentage of missing values per column."""

	total = len(df)
	rows: List[dict] = []

	for col in df.columns:
		missing_count = int(df[col].isna().sum())
		missing_pct = float(missing_count / total * 100) if total > 0 else 0.0
		rows.append(
			{
				"scope": "overall",
				"entity_level": "column",
				"entity_id": col,
				"entity_name": col,
				"metric": "missing_pct",
				"value": missing_pct,
			}
		)

	return pd.DataFrame(rows)


def compute_defect_rate_by_assembly_line(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute defect rate per assembly line."""

	if "defect_flag" not in df.columns or "assembly_line" not in df.columns:
		return pd.DataFrame()

	grouped = df.groupby("assembly_line")["defect_flag"].mean().reset_index()
	grouped["defect_rate_pct"] = grouped["defect_flag"] * 100

	rows = [
		{
			"scope": "assembly_line",
			"entity_level": "assembly_line",
			"entity_id": int(row["assembly_line"]),
			"entity_name": f"Assembly Line {int(row['assembly_line'])}",
			"metric": "defect_rate_pct",
			"value": float(row["defect_rate_pct"]),
		}
		for _, row in grouped.iterrows()
	]

	return pd.DataFrame(rows)


def compute_avg_torque_by_machine(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute average torque per machine."""

	required_cols = {"machine_id", "assembly_line", "machine_description", "torque_value"}
	if not required_cols.issubset(df.columns):
		return pd.DataFrame()

	grouped = (
		df.groupby(["machine_id", "assembly_line", "machine_description"])[
			"torque_value"
		]
		.mean()
		.reset_index()
		.rename(columns={"torque_value": "avg_torque"})
	)

	return grouped


def flag_torque_drift_machines(
	machine_torque: pd.DataFrame,
	z_threshold: float = 3.0,
) -> pd.DataFrame:
	"""Flag machines with abnormal torque drift using Z-score.

	Parameters
	----------
	machine_torque:
		DataFrame with columns: machine_id, assembly_line, machine_description, avg_torque
	z_threshold:
		Absolute Z-score above which a machine is considered abnormal.
	"""

	if machine_torque.empty:
		return pd.DataFrame()

	values = machine_torque["avg_torque"].astype(float)
	mean = values.mean()
	std = values.std(ddof=0)

	if std == 0 or np.isnan(std):
		LOGGER.warning("No variation in average torque; torque drift cannot be assessed")
		machine_torque["torque_zscore"] = 0.0
		machine_torque["torque_drift_flag"] = False
		return machine_torque

	z_scores = (values - mean) / std
	machine_torque["torque_zscore"] = z_scores
	machine_torque["torque_drift_flag"] = z_scores.abs() > z_threshold

	return machine_torque


def build_summary_report(
	df: pd.DataFrame,
	missing_pct: pd.DataFrame,
	defect_rates: pd.DataFrame,
	machine_drift: pd.DataFrame,
) -> pd.DataFrame:
	"""Combine all metrics into a long-form summary report."""

	parts: List[pd.DataFrame] = []

	if not missing_pct.empty:
		parts.append(missing_pct)

	if not defect_rates.empty:
		parts.append(defect_rates)

	if not machine_drift.empty:
		rows = [
			{
				"scope": "machine",
				"entity_level": "machine",
				"entity_id": int(row["machine_id"]),
				"entity_name": str(row["machine_description"]),
				"metric": "avg_torque",
				"value": float(row["avg_torque"]),
			}
			for _, row in machine_drift.iterrows()
		]

		z_rows = [
			{
				"scope": "machine",
				"entity_level": "machine",
				"entity_id": int(row["machine_id"]),
				"entity_name": str(row["machine_description"]),
				"metric": "torque_zscore",
				"value": float(row["torque_zscore"]),
			}
			for _, row in machine_drift.iterrows()
		]

		flag_rows = [
			{
				"scope": "machine",
				"entity_level": "machine",
				"entity_id": int(row["machine_id"]),
				"entity_name": str(row["machine_description"]),
				"metric": "torque_drift_flag",
				"value": bool(row["torque_drift_flag"]),
			}
			for _, row in machine_drift.iterrows()
		]

		parts.extend([pd.DataFrame(rows), pd.DataFrame(z_rows), pd.DataFrame(flag_rows)])

	if not parts:
		LOGGER.warning("No metrics were computed; summary report will be empty")
		return pd.DataFrame(columns=["scope", "entity_level", "entity_id", "entity_name", "metric", "value"])

	report = pd.concat(parts, ignore_index=True)
	return report


def save_report(report: pd.DataFrame, output_path: Path) -> None:
	"""Save the summary report as a CSV file."""

	output_path.parent.mkdir(parents=True, exist_ok=True)
	report.to_csv(output_path, index=False)
	LOGGER.info("Data quality report written to %s", output_path)


def main() -> None:
	configure_logging()
	LOGGER.info("Starting data quality checks")

	config = DbConfig()
	engine = get_engine(config)

	try:
		df = extract_data(engine)
	except Exception as exc:  # noqa: BLE001
		LOGGER.exception("Failed to extract data: %s", exc)
		raise

	missing_pct = compute_missing_percentages(df)
	defect_rates = compute_defect_rate_by_assembly_line(df)
	machine_torque = compute_avg_torque_by_machine(df)
	machine_drift = flag_torque_drift_machines(machine_torque)

	report = build_summary_report(df, missing_pct, defect_rates, machine_drift)

	output_path = Path(__file__).with_name("data_quality_report.csv")
	save_report(report, output_path)

	LOGGER.info("Data quality checks completed")


if __name__ == "__main__":
	main()

