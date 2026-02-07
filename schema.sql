-- Star schema for manufacturing data warehouse (PostgreSQL)

-- Drop tables if they already exist to allow re-creation during development
DROP TABLE IF EXISTS FactProduction CASCADE;
DROP TABLE IF EXISTS DimMachine CASCADE;
DROP TABLE IF EXISTS DimOperator CASCADE;
DROP TABLE IF EXISTS DimDate CASCADE;

-- Dimension: Date
CREATE TABLE DimDate (
	date_id      INTEGER PRIMARY KEY,          -- e.g., 20260101
	full_date    DATE      NOT NULL,
	day          SMALLINT  NOT NULL CHECK (day BETWEEN 1 AND 31),
	month        SMALLINT  NOT NULL CHECK (month BETWEEN 1 AND 12),
	year         SMALLINT  NOT NULL,
	week_number  SMALLINT  NOT NULL CHECK (week_number BETWEEN 1 AND 53)
);

-- Dimension: Machine
CREATE TABLE DimMachine (
	machine_id          SERIAL PRIMARY KEY,
	assembly_line       SMALLINT NOT NULL CHECK (assembly_line BETWEEN 1 AND 5),
	machine_description TEXT
);

-- Dimension: Operator
CREATE TABLE DimOperator (
	operator_id      INTEGER PRIMARY KEY,
	shift            VARCHAR(20) NOT NULL,   -- e.g., 'Day', 'Evening', 'Night'
	experience_level VARCHAR(20) NOT NULL    -- e.g., 'Junior', 'Mid', 'Senior'
);

-- Fact: Production
CREATE TABLE FactProduction (
	production_id        BIGSERIAL PRIMARY KEY,
	engine_id            UUID        NOT NULL,
	machine_id           INTEGER     NOT NULL,
	operator_id          INTEGER     NOT NULL,
	date_id              INTEGER     NOT NULL,
	torque_value         NUMERIC(10,2) NOT NULL,
	temperature_celsius  NUMERIC(10,2) NOT NULL,
	cycle_time_seconds   NUMERIC(10,2) NOT NULL,
	defect_flag          BOOLEAN     NOT NULL,
	downtime_minutes     NUMERIC(10,2) NOT NULL DEFAULT 0,

	CONSTRAINT fk_factproduction_machine
		FOREIGN KEY (machine_id)
		REFERENCES DimMachine (machine_id),

	CONSTRAINT fk_factproduction_operator
		FOREIGN KEY (operator_id)
		REFERENCES DimOperator (operator_id),

	CONSTRAINT fk_factproduction_date
		FOREIGN KEY (date_id)
		REFERENCES DimDate (date_id)
);
