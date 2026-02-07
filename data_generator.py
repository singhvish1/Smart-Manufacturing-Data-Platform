"""Data generator for simulated engine production manufacturing data.

This script generates 10,000 records of engine production data and saves
them to a CSV file named ``raw_production_data.csv`` in the same directory
as this script. It is intended for use in analytics, anomaly detection,
and dashboarding demos.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd


NUM_RECORDS = 10_000
DEFECT_RATE = 0.05
DOWNTIME_PROBABILITY = 0.10
MIN_DOWNTIME_MINUTES = 5
MAX_DOWNTIME_MINUTES = 60


def _align_to_shift(current_time: pd.Timestamp) -> pd.Timestamp:
	"""Align a timestamp to active production shifts.

	We simulate two 8-hour shifts per day:
	- Day shift: 06:00–14:00
	- Evening shift: 14:00–22:00

	If the current time is outside these windows, move it to the next
	shift start time.
	"""

	hour = current_time.hour
	date = current_time.normalize()

	# Already within an active shift
	if 6 <= hour < 14:
		return current_time
	if 14 <= hour < 22:
		return current_time

	# Before shifts start: jump to 06:00 same day
	if hour < 6:
		return date + pd.Timedelta(hours=6)

	# After 22:00: jump to 06:00 next day
	return date + pd.Timedelta(days=1, hours=6)


def generate_production_data(num_records: int = NUM_RECORDS) -> pd.DataFrame:
	"""Generate simulated engine production data.

	Parameters
	----------
	num_records:
		Number of production records to generate.
	"""

	# For reproducibility in demos; comment this out if you
	# prefer different random data on each run.
	np.random.seed(42)

	# Basic categorical / id-like fields
	engine_ids = [str(uuid.uuid4()) for _ in range(num_records)]
	assembly_lines = np.random.randint(1, 6, size=num_records)  # 1–5 inclusive
	operator_ids = np.random.randint(100, 151, size=num_records)  # 100–150 inclusive

	# Process metrics (clipped to avoid unrealistic negatives)
	torque_values = np.random.normal(loc=500.0, scale=15.0, size=num_records)
	temperature_celsius = np.random.normal(loc=90.0, scale=5.0, size=num_records)
	cycle_time_seconds = np.random.normal(loc=300.0, scale=30.0, size=num_records)

	# Ensure non-negative, somewhat realistic values
	torque_values = torque_values.clip(min=400.0)
	temperature_celsius = temperature_celsius.clip(min=60.0)
	cycle_time_seconds = cycle_time_seconds.clip(min=60.0)

	# Defects: 5% defect rate
	defect_flags = np.random.rand(num_records) < DEFECT_RATE

	# Downtime: mostly 0, but 10% chance of 5–60 minutes
	downtime_minutes = np.zeros(num_records, dtype=float)
	downtime_mask = np.random.rand(num_records) < DOWNTIME_PROBABILITY
	downtime_minutes[downtime_mask] = np.random.uniform(
		MIN_DOWNTIME_MINUTES,
		MAX_DOWNTIME_MINUTES,
		size=downtime_mask.sum(),
	)

	# Timestamps: realistic sequential production over multiple days
	# Start at a fixed date for reproducibility
	current_time = pd.Timestamp("2026-01-01 06:00:00")
	timestamps: list[pd.Timestamp] = []

	for i in range(num_records):
		current_time = _align_to_shift(current_time)

		duration_seconds = float(cycle_time_seconds[i]) + float(downtime_minutes[i]) * 60.0
		current_time = current_time + pd.Timedelta(seconds=duration_seconds)
		timestamps.append(current_time)

	# Build DataFrame
	df = pd.DataFrame(
		{
			"engine_id": engine_ids,
			"assembly_line": assembly_lines,
			"operator_id": operator_ids,
			"torque_value": torque_values,
			"temperature_celsius": temperature_celsius,
			"cycle_time_seconds": cycle_time_seconds,
			"defect_flag": defect_flags,
			"downtime_minutes": downtime_minutes,
			"timestamp": timestamps,
		}
	)

	return df


def main() -> None:
	"""Generate data and write to raw_production_data.csv."""

	df = generate_production_data(NUM_RECORDS)

	output_path = Path(__file__).with_name("raw_production_data.csv")
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df.to_csv(output_path, index=False)
	print(f"Generated {len(df)} records to {output_path}")


if __name__ == "__main__":
	main()
