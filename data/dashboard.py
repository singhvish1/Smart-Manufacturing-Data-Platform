"""Streamlit dashboard for manufacturing production analytics.

This app reads aggregated data from a PostgreSQL data warehouse populated
by etl_pipeline.py and visualizes key production metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class DbConfig:
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


@st.cache_resource(show_spinner=False)
def get_engine(config: DbConfig) -> Engine:
	"""Create and cache a SQLAlchemy engine."""

	return create_engine(config.sqlalchemy_url(), future=True)


@st.cache_data(show_spinner=False)
def load_data(_engine: Engine) -> pd.DataFrame:
	"""Load joined fact/dimension data for analytics.

	Joins FactProduction with DimMachine and DimDate.
	"""

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

	with _engine.connect() as conn:
		df = pd.read_sql(query, conn)

	df["full_date"] = pd.to_datetime(df["full_date"])
	return df


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp]]:
	"""Render sidebar filters and return filtered DataFrame and date range."""

	st.sidebar.header("Filters")

	min_date = df["full_date"].min()
	max_date = df["full_date"].max()

	start_date, end_date = st.sidebar.date_input(
		"Date range",
		value=(min_date, max_date),
		min_value=min_date,
		max_value=max_date,
	)

	# Ensure the widget returns a tuple
	if not isinstance(start_date, pd.Timestamp):
		start_date = pd.to_datetime(start_date)
	if not isinstance(end_date, pd.Timestamp):
		end_date = pd.to_datetime(end_date)

	assembly_lines = sorted(df["assembly_line"].unique().tolist())
	selected_lines = st.sidebar.multiselect(
		"Assembly line",
		options=assembly_lines,
		default=assembly_lines,
	)

	mask = (df["full_date"] >= start_date) & (df["full_date"] <= end_date)
	if selected_lines:
		mask &= df["assembly_line"].isin(selected_lines)

	df_filtered = df.loc[mask].copy()
	return df_filtered, (start_date, end_date)


def render_kpis(df: pd.DataFrame) -> None:
	"""Render KPI cards at the top of the dashboard."""

	total_engines = int(len(df))
	defect_rate = float(df["defect_flag"].mean() * 100) if total_engines > 0 else 0.0
	avg_cycle_time = float(df["cycle_time_seconds"].mean()) if total_engines > 0 else 0.0
	total_downtime = float(df["downtime_minutes"].sum())

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Total engines produced", f"{total_engines:,}")
	col2.metric("Defect rate", f"{defect_rate:.2f}%")
	col3.metric("Average cycle time (s)", f"{avg_cycle_time:.1f}")
	col4.metric("Total downtime (min)", f"{total_downtime:.1f}")


def render_charts(df: pd.DataFrame) -> None:
	"""Render line, bar, and histogram charts."""

	if df.empty:
		st.warning("No data available for the selected filters.")
		return

	# Line chart: torque trend over time (daily average)
	torque_trend = (
		df.groupby("full_date", as_index=False)["torque_value"].mean().rename(
			columns={"torque_value": "avg_torque"}
		)
	)

	st.subheader("Torque trend over time")
	st.line_chart(torque_trend.set_index("full_date")["avg_torque"])

	# Bar chart: defects per assembly line
	defects_by_line = (
		df.groupby("assembly_line")["defect_flag"].mean().reset_index()
	)
	defects_by_line["defect_rate"] = defects_by_line["defect_flag"] * 100

	st.subheader("Defect rate by assembly line")
	st.bar_chart(
		data=defects_by_line.set_index("assembly_line")["defect_rate"],
	)

	# Histogram: cycle time distribution
	st.subheader("Cycle time distribution")
	fig, ax = plt.subplots(figsize=(6, 3))
	ax.hist(df["cycle_time_seconds"], bins=30, color="#1f77b4", edgecolor="white")
	ax.set_xlabel("Cycle time (seconds)")
	ax.set_ylabel("Count")
	ax.set_title("Cycle time distribution")
	st.pyplot(fig)


def render_top_downtime_table(df: pd.DataFrame) -> None:
	"""Render table for top 5 machines with highest downtime."""

	if df.empty:
		return

	downtime_by_machine = (
		df.groupby(["machine_id", "machine_description"], as_index=False)[
			"downtime_minutes"
		]
		.sum()
		.sort_values("downtime_minutes", ascending=False)
		.head(5)
	)

	downtime_by_machine["downtime_minutes"] = downtime_by_machine[
		"downtime_minutes"
	].round(1)

	st.subheader("Top 5 machines by downtime (minutes)")
	st.table(downtime_by_machine)


def main() -> None:
	st.set_page_config(
		page_title="Manufacturing Production Dashboard",
		page_icon="⚙️",
		layout="wide",
	)

	st.title("Manufacturing Production Analytics")
	st.caption("Interactive overview of engine production performance")

	config = DbConfig()

	try:
		engine = get_engine(config)
		df = load_data(engine)
	except Exception as exc:  # noqa: BLE001
		st.error(f"Failed to connect to database or load data: {exc}")
		return

	df_filtered, date_range = apply_filters(df)

	render_kpis(df_filtered)
	st.markdown("---")
	render_charts(df_filtered)
	st.markdown("---")
	render_top_downtime_table(df_filtered)


if __name__ == "__main__":
	main()

