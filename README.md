# BigData benchmark

This repository contains my BigData homework project.  
The goal of the work is to compare different data formats and tools for storing, reading and querying data.

## What is compared

The project includes benchmarks for the following formats and tools:

- CSV
- TSV
- JSON
- JSONL
- Parquet
- ORC
- SQLite
- DuckDB
- Polars
- pandas / sqlite3
- json, orjson, ujson

## Main report

The main results are presented in:

`notebook_with_results.ipynb`

In the notebook I describe the SQL queries, show the benchmark tables and charts, and compare the results for CSV, Parquet and SQLite.

## SQL queries

Three types of SQL queries are used:

1. filtering;
2. grouping;
3. window function.

They are used to check how different formats and libraries handle not only simple reading, but also analytical operations.

## Results

The benchmark result files are stored in the `benchmarks/` folder.

Large generated data files are not included in this repository because they are too large for GitHub. The repository contains the notebook, scripts, result tables and query previews.
