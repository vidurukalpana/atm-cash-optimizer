-- EDA queries on the DuckDB warehouse (data/warehouse.duckdb).
-- Each query starts with "-- name: <name>"; load them with atm_cash_optimizer.io.read_sql_queries
-- and run them with atm_cash_optimizer.io.query (see notebooks/01_eda.ipynb).

-- name: weekday_averages
-- Mean withdrawals per weekday over all machines (isodow: 1 = Monday ... 7 = Sunday).
SELECT
    isodow(date)                   AS iso_weekday,
    dayname(date)                  AS weekday,
    round(avg(withdrawals_lkr))    AS avg_withdrawals_lkr
FROM nn5_daily
GROUP BY ALL
ORDER BY iso_weekday;

-- name: monthly_totals
-- Total withdrawals of the whole fleet per calendar month.
SELECT
    date_trunc('month', date)::DATE   AS month,
    count(DISTINCT date)              AS days,
    round(sum(withdrawals_lkr))       AS total_withdrawals_lkr
FROM nn5_daily
GROUP BY ALL
ORDER BY month;

-- name: top_volatility_machines
-- Machines with the highest coefficient of variation (std / mean of daily withdrawals).
SELECT
    machine_id,
    round(avg(withdrawals_lkr))                              AS avg_withdrawals_lkr,
    round(stddev_samp(withdrawals_lkr) / avg(withdrawals_lkr), 3) AS cv
FROM nn5_daily
GROUP BY machine_id
ORDER BY cv DESC
LIMIT 10;

-- name: month_end_vs_mid_month
-- Mean withdrawals in the last 3 and first 3 days of a month vs the rest of the month.
SELECT
    CASE
        WHEN day(date) <= 3 OR day(date) > day(last_day(date)) - 3 THEN 'month_end'
        ELSE 'mid_month'
    END                            AS period,
    count(*)                       AS machine_days,
    round(avg(withdrawals_lkr))    AS avg_withdrawals_lkr
FROM nn5_daily
GROUP BY ALL
ORDER BY period;
