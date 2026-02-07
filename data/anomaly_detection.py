"""Production anomaly detection module.

This module:
- Loads production data from the PostgreSQL warehouse (FactProduction and dims)
- Detects anomalies using:
  - Z-score for ``torque_value``
  - Isolation Forest for multivariate anomalies
- Flags anomalous production records and stores them in ``ProductionAnomalies``
- Generates a visualization of anomaly counts over time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

	LOGGER.info("Creating SQLAlchemy engine for anomaly detection")
	return create_engine(config.sqlalchemy_url(), future=True)


def extract_production_data(engine: Engine) -> pd.DataFrame:
	"""Extract production data joined with date and machine dimensions."""

	LOGGER.info("Extracting production data from warehouse")

	query = text(
		"""
		SELECT
			fp.production_id,
			fp.engine_id,
			fp.machine_id,
			m.assembly_line,
			m.machine_description,
			fp.operator_id,
			d.date_id,
			d.full_date,
			fp.torque_value,
			fp.temperature_celsius,
			fp.cycle_time_seconds,
			fp.defect_flag,
			fp.downtime_minutes
		FROM FactProduction fp
		JOIN DimMachine m ON fp.machine_id = m.machine_id
		JOIN DimDate d ON fp.date_id = d.date_id
		ORDER BY d.full_date, fp.production_id
		"""
	)

	with engine.connect() as conn:
		df = pd.read_sql(query, conn)

	if df.empty:
		LOGGER.warning("No production data found in warehouse")
	else:
		LOGGER.info("Loaded %d production rows", len(df))

	df["full_date"] = pd.to_datetime(df["full_date"])
	return df


def _detect_zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
	"""Detect anomalies based on Z-score of torque_value."""

	torque = df["torque_value"].astype(float)
	mean = torque.mean()
	std = torque.std(ddof=0)

	if std == 0 or np.isnan(std):
		LOGGER.warning("Torque standard deviation is zero; no Z-score anomalies will be flagged")
		return pd.Series(False, index=df.index, name="zscore_anomaly")

	z_scores = (torque - mean) / std
	z_anomaly = z_scores.abs() > threshold

	LOGGER.info("Z-score anomalies detected: %d", int(z_anomaly.sum()))
	df["torque_zscore"] = z_scores
	return z_anomaly.rename("zscore_anomaly")


def _detect_isolation_forest_anomalies(
	df: pd.DataFrame,
	contamination: float = 0.05,
	random_state: int = 42,
) -> Tuple[pd.Series, pd.Series]:
	"""Detect multivariate anomalies using Isolation Forest.

	Returns
	-------
	(iforest_anomaly, iforest_score)
	"""

	LOGGER.info("Running Isolation Forest for multivariate anomaly detection")

	feature_cols = [
		"torque_value",
		"temperature_celsius",
		"cycle_time_seconds",
		"downtime_minutes",
	]

	X = df[feature_cols].astype(float).copy()

	# Standardize features for better model performance
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	model = IsolationForest(
		contamination=contamination,
		random_state=random_state,
		n_estimators=200,
		n_jobs=-1,
	)

	model.fit(X_scaled)
	preds = model.predict(X_scaled)  # -1 anomaly, 1 normal
	scores = -model.decision_function(X_scaled)  # higher score => more anomalous

	anomaly_series = pd.Series(preds == -1, index=df.index, name="iforest_anomaly")
	score_series = pd.Series(scores, index=df.index, name="iforest_score")

	LOGGER.info("Isolation Forest anomalies detected: %d", int(anomaly_series.sum()))
	return anomaly_series, score_series


def transform_detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
	"""Run anomaly detection and return anomalies as a DataFrame.

	The returned DataFrame contains only anomalous records with flags and scores.
	"""

	if df.empty:
		LOGGER.warning("No data to run anomaly detection on")
		return pd.DataFrame()

	LOGGER.info("Starting anomaly detection transform step")

	z_anomaly = _detect_zscore_anomalies(df)
	if_anomaly, if_score = _detect_isolation_forest_anomalies(df)

	df["zscore_anomaly"] = z_anomaly
	df["iforest_anomaly"] = if_anomaly
	df["iforest_score"] = if_score

	# Derive consolidated anomaly type
	def _label_row(row: pd.Series) -> str:
		z = bool(row["zscore_anomaly"])
		iso = bool(row["iforest_anomaly"])
		if z and iso:
			return "both"
		if z:
			return "zscore"
		if iso:
			return "isolation_forest"
		return "none"

	df["anomaly_type"] = df.apply(_label_row, axis=1)

	anomalies = df[df["anomaly_type"] != "none"].copy()
	LOGGER.info("Total anomalous records: %d", len(anomalies))

	return anomalies


def _ensure_anomalies_table(engine: Engine) -> None:
	"""Create ProductionAnomalies table if it does not exist."""

	LOGGER.info("Ensuring ProductionAnomalies table exists")
	ddl = text(
		"""
		CREATE TABLE IF NOT EXISTS ProductionAnomalies (
			anomaly_id        BIGSERIAL PRIMARY KEY,
			production_id     BIGINT      NOT NULL,
			engine_id         UUID        NOT NULL,
			machine_id        INTEGER     NOT NULL,
			assembly_line     SMALLINT    NOT NULL,
			operator_id       INTEGER     NOT NULL,
			date_id           INTEGER     NOT NULL,
			full_date         DATE        NOT NULL,
			torque_value      NUMERIC(10,2) NOT NULL,
			temperature_celsius NUMERIC(10,2) NOT NULL,
			cycle_time_seconds  NUMERIC(10,2) NOT NULL,
			defect_flag       BOOLEAN     NOT NULL,
			downtime_minutes  NUMERIC(10,2) NOT NULL,
			torque_zscore     NUMERIC(12,4),
			iforest_score     NUMERIC(12,6),
			zscore_anomaly    BOOLEAN     NOT NULL,
			iforest_anomaly   BOOLEAN     NOT NULL,
			anomaly_type      VARCHAR(32) NOT NULL,
			detected_at       TIMESTAMP   NOT NULL DEFAULT NOW(),
			CONSTRAINT fk_anom_production
				FOREIGN KEY (production_id)
				REFERENCES FactProduction (production_id)
		);
		"""
	)

	with engine.begin() as conn:
		conn.execute(ddl)


def load_anomalies(engine: Engine, anomalies: pd.DataFrame) -> None:
	"""Persist anomalies to the ProductionAnomalies table."""

	if anomalies.empty:
		LOGGER.info("No anomalies to load. Skipping DB insert.")
		return

	_ensure_anomalies_table(engine)

	LOGGER.info("Loading %d anomalies into ProductionAnomalies", len(anomalies))

	# Columns to persist
	cols = [
		"production_id",
		"engine_id",
		"machine_id",
		"assembly_line",
		"operator_id",
		"date_id",
		"full_date",
		"torque_value",
		"temperature_celsius",
		"cycle_time_seconds",
		"defect_flag",
		"downtime_minutes",
		"torque_zscore",
		"iforest_score",
		"zscore_anomaly",
		"iforest_anomaly",
		"anomaly_type",
	]

	records = anomalies[cols].to_dict(orient="records")

	insert_sql = text(
		"""
		INSERT INTO ProductionAnomalies (
			production_id,
			engine_id,
			machine_id,
			assembly_line,
			operator_id,
			date_id,
			full_date,
			torque_value,
			temperature_celsius,
			cycle_time_seconds,
			defect_flag,
			downtime_minutes,
			torque_zscore,
			iforest_score,
			zscore_anomaly,
			iforest_anomaly,
			anomaly_type
		) VALUES (
			:production_id,
			:engine_id,
			:machine_id,
			:assembly_line,
			:operator_id,
			:date_id,
			:full_date,
			:torque_value,
			:temperature_celsius,
			:cycle_time_seconds,
			:defect_flag,
			:downtime_minutes,
			:torque_zscore,
			:iforest_score,
			:zscore_anomaly,
			:iforest_anomaly,
			:anomaly_type
		);
		"""
	)

	with engine.begin() as conn:
		conn.execute(insert_sql, records)


def visualize_anomaly_counts(anomalies: pd.DataFrame, output_path: Path | None = None) -> Path | None:
	"""Create a plot of anomaly counts over time and save to disk.

	Returns the path to the saved PNG (or None if no data).
	"""

	if anomalies.empty:
		LOGGER.warning("No anomalies available for visualization")
		return None

	LOGGER.info("Creating anomaly counts over time visualization")

	counts = (
		anomalies.groupby("full_date")
		.size()
		.reset_index(name="anomaly_count")
		.sort_values("full_date")
	)

	fig, ax = plt.subplots(figsize=(8, 4))
	ax.plot(counts["full_date"], counts["anomaly_count"], marker="o")
	ax.set_xlabel("Date")
	ax.set_ylabel("Anomaly count")
	ax.set_title("Production anomaly counts over time")
	fig.autofmt_xdate()

	if output_path is None:
		output_path = Path(__file__).with_name("anomaly_counts_over_time.png")

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)

	LOGGER.info("Saved anomaly counts plot to %s", output_path)
	return output_path


def main() -> None:
	configure_logging()
	LOGGER.info("Starting anomaly detection pipeline")

	config = DbConfig()
	engine = get_engine(config)

	df = extract_production_data(engine)
	anomalies = transform_detect_anomalies(df)
	load_anomalies(engine, anomalies)
	visualize_anomaly_counts(anomalies)

	LOGGER.info("Anomaly detection pipeline completed")


if __name__ == "__main__":
	main()

