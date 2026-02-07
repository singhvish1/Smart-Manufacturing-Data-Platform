"""Streamlit dashboard for manufacturing production analytics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class DbConfig:
	"""Database configuration using environment variables for Streamlit app."""

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
def get_engine() -> Engine:
	"""Create and cache a SQLAlchemy engine for PostgreSQL."""

	config = DbConfig()
	engine = create_engine(config.sqlalchemy_url(), future=True)
	return engine


@st.cache_data(show_spinner=True)
def load_data(_engine: Engine) -> pd.DataFrame:
	"""Load joined production and dimension data into a DataFrame."""

	query = text(
		"""
		SELECT
			fp.production_id,
			fp.engine_id,
			fp.machine_id,
			m.assembly_line,
			m.machine_description,
			fp.operator_id,
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

	if df.empty:
		return df

	df["full_date"] = pd.to_datetime(df["full_date"]).dt.date
	return df


def apply_filters(
	df: pd.DataFrame,
	start_date: Optional[date],
	end_date: Optional[date],
	assembly_line: Optional[int],
) -> pd.DataFrame:
	"""Filter data by date range and assembly line."""

	filtered = df.copy()

	if start_date is not None:
		filtered = filtered[filtered["full_date"] >= start_date]
	if end_date is not None:
		filtered = filtered[filtered["full_date"] <= end_date]
	if assembly_line is not None and assembly_line != -1:
		filtered = filtered[filtered["assembly_line"] == assembly_line]

	return filtered


def render_kpis(df: pd.DataFrame) -> None:
	"""Render high-level KPI metrics."""

	if df.empty:
		st.info("No data available for the selected filters.")
		return

	total_units = len(df)
	defect_rate = df["defect_flag"].fillna(False).mean() * 100.0
	avg_cycle_time = df["cycle_time_seconds"].mean()
	total_downtime = df["downtime_minutes"].sum()

	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Total Units", f"{total_units:,}")
	with col2:
		st.metric("Defect Rate", f"{defect_rate:.2f}%")
	with col3:
		st.metric("Avg Cycle Time (s)", f"{avg_cycle_time:.1f}")
	with col4:
		st.metric("Total Downtime (min)", f"{total_downtime:.1f}")


def render_charts(df: pd.DataFrame) -> None:
	"""Render charts: torque trend, defects by line, cycle-time distribution."""

	if df.empty:
		return

	st.subheader("Torque over time")
	torque_by_date = (
		df.groupby("full_date")["torque_value"]
		.mean()
		.reset_index()
		.sort_values("full_date")
	)
	st.line_chart(torque_by_date, x="full_date", y="torque_value")

	st.subheader("Defects by assembly line")
	defects_by_line = (
		df.groupby("assembly_line")["defect_flag"]
		.mean()
		.reset_index()
	)
	defects_by_line["defect_rate_pct"] = defects_by_line["defect_flag"] * 100.0
	st.bar_chart(defects_by_line, x="assembly_line", y="defect_rate_pct")

	st.subheader("Cycle time distribution")
	fig, ax = plt.subplots()
	ax.hist(df["cycle_time_seconds"].dropna(), bins=20, edgecolor="black")
	ax.set_xlabel("Cycle time (seconds)")
	ax.set_ylabel("Frequency")
	st.pyplot(fig)


def render_top_downtime_table(df: pd.DataFrame, top_n: int = 5) -> None:
	"""Render table of machines with the highest downtime."""

	if df.empty:
		return

	st.subheader(f"Top {top_n} machines by downtime")
	downtime_by_machine = (
		df.groupby(["machine_id", "machine_description"])["downtime_minutes"]
		.sum()
		.reset_index()
		.sort_values("downtime_minutes", ascending=False)
		.head(top_n)
	)
	st.dataframe(downtime_by_machine, use_container_width=True)


def main() -> None:
	st.set_page_config(page_title="Manufacturing Production Dashboard", layout="wide")
	st.title("Manufacturing Production Analytics")

	engine = get_engine()

	with st.spinner("Loading data from warehouse..."):
		df = load_data(engine)

	if df.empty:
		st.warning("No data found in FactProduction. Run the ETL pipeline first.")
		return

	# Sidebar filters
	st.sidebar.header("Filters")
	min_date = df["full_date"].min()
	max_date = df["full_date"].max()

	start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
	end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

	assembly_lines = sorted(df["assembly_line"].unique().tolist())
	assembly_line_labels = ["All lines"] + [f"Line {al}" for al in assembly_lines]
	selected_idx = st.sidebar.selectbox("Assembly line", options=list(range(len(assembly_line_labels))), format_func=lambda i: assembly_line_labels[i])
	selected_line = None if selected_idx == 0 else assembly_lines[selected_idx - 1]

	filtered = apply_filters(df, start_date, end_date, selected_line)

	render_kpis(filtered)
	render_charts(filtered)
	render_top_downtime_table(filtered)


if __name__ == "__main__":
	main()
