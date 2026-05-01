import gc
import os
import time
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

# =========================
# 路径设置
# =========================
BASE_DIR = Path(".")
CSV_FILE = BASE_DIR / "data_formats" / "csv" / "docs.csv"
PARQUET_FILE = BASE_DIR / "data_formats" / "parquet" / "docs.parquet"
SQLITE_FILE = BASE_DIR / "data_formats" / "sqlite" / "docs.db"

OUTPUT_DIR = BASE_DIR / "benchmarks"
QUERY_OUTPUT_DIR = OUTPUT_DIR / "query_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "benchmark_results.csv"
RESULT_TXT = OUTPUT_DIR / "benchmark_results.txt"

REPEAT = 3
SQLITE_TABLE = "docs"

# =========================
# 基础工具
# =========================
def benchmark(func, repeat=3):
    cpu_times = []
    wall_times = []
    last_result = None

    for _ in range(repeat):
        gc.collect()

        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        last_result = func()

        cpu_times.append(time.process_time() - t0_cpu)
        wall_times.append(time.perf_counter() - t0_wall)

    return {
        "cpu_time_avg_sec": sum(cpu_times) / len(cpu_times),
        "wall_time_avg_sec": sum(wall_times) / len(wall_times),
        "cpu_time_min_sec": min(cpu_times),
        "wall_time_min_sec": min(wall_times),
        "cpu_time_max_sec": max(cpu_times),
        "wall_time_max_sec": max(wall_times),
        "last_result": last_result,
    }


def sizeof_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def ensure_inputs():
    missing = [str(p) for p in [CSV_FILE, PARQUET_FILE, SQLITE_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))


# =========================
# 读取 benchmark
# =========================
def read_csv_polars():
    df = pl.read_csv(CSV_FILE)
    return df.height, df.width


def read_parquet_polars():
    df = pl.read_parquet(PARQUET_FILE)
    return df.height, df.width


def read_sqlite_pandas():
    with sqlite3.connect(SQLITE_FILE) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {SQLITE_TABLE}", conn)
    return df.shape[0], df.shape[1]


# =========================
# SQL 查询：DuckDB
# =========================
def run_duck_query_on_csv(sql: str):
    con = duckdb.connect(database=":memory:")
    try:
        escaped = CSV_FILE.as_posix().replace("'", "''")
        con.execute(f"""
            CREATE VIEW docs AS
            SELECT *
            FROM read_csv_auto('{escaped}', header=True, sample_size=-1);
        """)
        result = con.execute(sql).fetchdf()
        return result
    finally:
        con.close()


def run_duck_query_on_parquet(sql: str):
    con = duckdb.connect(database=":memory:")
    try:
        escaped = PARQUET_FILE.as_posix().replace("'", "''")
        con.execute(f"""
            CREATE VIEW docs AS
            SELECT *
            FROM read_parquet('{escaped}');
        """)
        result = con.execute(sql).fetchdf()
        return result
    finally:
        con.close()


# =========================
# SQL 查询：SQLite
# =========================
def run_sqlite_query(sql: str):
    with sqlite3.connect(SQLITE_FILE) as conn:
        result = pd.read_sql_query(sql, conn)
    return result


# =========================
# benchmark 用 SQL
# 这些查询会尽量保证真正执行，不只是看几行
# =========================
QUERY_FILTER_BENCH = """
SELECT COUNT(*) AS matched_rows,
       AVG(text_len) AS avg_text_len
FROM docs
WHERE domain <> ''
  AND text_len > 1000;
"""

QUERY_GROUP_BENCH = """
SELECT domain,
       COUNT(*) AS doc_cnt,
       AVG(text_len) AS avg_text_len
FROM docs
WHERE domain <> ''
GROUP BY domain
ORDER BY doc_cnt DESC, domain
LIMIT 20;
"""

QUERY_WINDOW_BENCH = """
SELECT COUNT(*) AS n_rows,
       MAX(rn) AS max_rn
FROM (
    SELECT ROW_NUMBER() OVER (
               PARTITION BY domain
               ORDER BY warc_date
           ) AS rn
    FROM docs
    WHERE domain <> ''
) t;
"""

# =========================
# 预览用 SQL
# 这些结果会保存到 query_outputs 文件夹
# =========================
QUERY_FILTER_PREVIEW = """
SELECT doc_id, domain, warc_date, text_len
FROM docs
WHERE domain <> ''
  AND text_len > 1000
LIMIT 20;
"""

QUERY_GROUP_PREVIEW = """
SELECT domain,
       COUNT(*) AS doc_cnt,
       AVG(text_len) AS avg_text_len
FROM docs
WHERE domain <> ''
GROUP BY domain
ORDER BY doc_cnt DESC, domain
LIMIT 20;
"""

QUERY_WINDOW_PREVIEW = """
SELECT domain, warc_date, text_len,
       ROW_NUMBER() OVER (
           PARTITION BY domain
           ORDER BY warc_date
       ) AS rn
FROM docs
WHERE domain <> ''
LIMIT 50;
"""

# =========================
# 保存查询预览
# =========================
def save_preview(df, out_path: Path):
    if isinstance(df, pd.DataFrame):
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(df).to_csv(out_path, index=False, encoding="utf-8-sig")


# =========================
# 主逻辑
# =========================
def main():
    ensure_inputs()

    rows = []

    # -------------------------
    # 文件大小记录
    # -------------------------
    rows.append({
        "category": "file_size",
        "format": "csv",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{sizeof_mb(CSV_FILE):.2f} MB",
    })

    rows.append({
        "category": "file_size",
        "format": "parquet",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{sizeof_mb(PARQUET_FILE):.2f} MB",
    })

    rows.append({
        "category": "file_size",
        "format": "sqlite",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{sizeof_mb(SQLITE_FILE):.2f} MB",
    })

    # -------------------------
    # 读取 benchmark
    # -------------------------
    print("1) 读取 benchmark ...")

    r = benchmark(read_csv_polars, repeat=REPEAT)
    rows.append({
        "category": "read",
        "format": "csv",
        "engine": "polars",
        "task": "full_read",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": f"rows={r['last_result'][0]}, cols={r['last_result'][1]}",
    })

    r = benchmark(read_parquet_polars, repeat=REPEAT)
    rows.append({
        "category": "read",
        "format": "parquet",
        "engine": "polars",
        "task": "full_read",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": f"rows={r['last_result'][0]}, cols={r['last_result'][1]}",
    })

    r = benchmark(read_sqlite_pandas, repeat=REPEAT)
    rows.append({
        "category": "read",
        "format": "sqlite",
        "engine": "pandas+sqlite3",
        "task": "full_read",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": f"rows={r['last_result'][0]}, cols={r['last_result'][1]}",
    })

    # -------------------------
    # 查询 benchmark: CSV
    # -------------------------
    print("2) CSV 查询 benchmark ...")

    r = benchmark(lambda: run_duck_query_on_csv(QUERY_FILTER_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "csv",
        "engine": "duckdb",
        "task": "filter",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": r["last_result"].to_dict(orient="records")[0],
    })

    r = benchmark(lambda: run_duck_query_on_csv(QUERY_GROUP_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "csv",
        "engine": "duckdb",
        "task": "group_by",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": f"rows={len(r['last_result'])}",
    })

    r = benchmark(lambda: run_duck_query_on_csv(QUERY_WINDOW_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "csv",
        "engine": "duckdb",
        "task": "window",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": r["last_result"].to_dict(orient="records")[0],
    })

    # 保存 CSV 查询预览
    save_preview(run_duck_query_on_csv(QUERY_FILTER_PREVIEW), QUERY_OUTPUT_DIR / "csv_filter_preview.csv")
    save_preview(run_duck_query_on_csv(QUERY_GROUP_PREVIEW), QUERY_OUTPUT_DIR / "csv_group_preview.csv")
    save_preview(run_duck_query_on_csv(QUERY_WINDOW_PREVIEW), QUERY_OUTPUT_DIR / "csv_window_preview.csv")

    # -------------------------
    # 查询 benchmark: Parquet
    # -------------------------
    print("3) Parquet 查询 benchmark ...")

    r = benchmark(lambda: run_duck_query_on_parquet(QUERY_FILTER_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "parquet",
        "engine": "duckdb",
        "task": "filter",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": r["last_result"].to_dict(orient="records")[0],
    })

    r = benchmark(lambda: run_duck_query_on_parquet(QUERY_GROUP_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "parquet",
        "engine": "duckdb",
        "task": "group_by",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": f"rows={len(r['last_result'])}",
    })

    r = benchmark(lambda: run_duck_query_on_parquet(QUERY_WINDOW_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "parquet",
        "engine": "duckdb",
        "task": "window",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": r["last_result"].to_dict(orient="records")[0],
    })

    # 保存 Parquet 查询预览
    save_preview(run_duck_query_on_parquet(QUERY_FILTER_PREVIEW), QUERY_OUTPUT_DIR / "parquet_filter_preview.csv")
    save_preview(run_duck_query_on_parquet(QUERY_GROUP_PREVIEW), QUERY_OUTPUT_DIR / "parquet_group_preview.csv")
    save_preview(run_duck_query_on_parquet(QUERY_WINDOW_PREVIEW), QUERY_OUTPUT_DIR / "parquet_window_preview.csv")

    # -------------------------
    # 查询 benchmark: SQLite
    # -------------------------
    print("4) SQLite 查询 benchmark ...")

    r = benchmark(lambda: run_sqlite_query(QUERY_FILTER_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "sqlite",
        "engine": "sqlite3",
        "task": "filter",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": r["last_result"].to_dict(orient="records")[0],
    })

    r = benchmark(lambda: run_sqlite_query(QUERY_GROUP_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "sqlite",
        "engine": "sqlite3",
        "task": "group_by",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": f"rows={len(r['last_result'])}",
    })

    r = benchmark(lambda: run_sqlite_query(QUERY_WINDOW_BENCH), repeat=REPEAT)
    rows.append({
        "category": "query",
        "format": "sqlite",
        "engine": "sqlite3",
        "task": "window",
        **{k: v for k, v in r.items() if k != "last_result"},
        "result_info": r["last_result"].to_dict(orient="records")[0],
    })

    # 保存 SQLite 查询预览
    save_preview(run_sqlite_query(QUERY_FILTER_PREVIEW), QUERY_OUTPUT_DIR / "sqlite_filter_preview.csv")
    save_preview(run_sqlite_query(QUERY_GROUP_PREVIEW), QUERY_OUTPUT_DIR / "sqlite_group_preview.csv")
    save_preview(run_sqlite_query(QUERY_WINDOW_PREVIEW), QUERY_OUTPUT_DIR / "sqlite_window_preview.csv")

    # -------------------------
    # 输出结果表
    # -------------------------
    result_df = pd.DataFrame(rows)

    # 排序方便看
    category_order = {"file_size": 0, "read": 1, "query": 2}
    task_order = {"file_size_mb": 0, "full_read": 1, "filter": 2, "group_by": 3, "window": 4}

    result_df["_category_order"] = result_df["category"].map(category_order)
    result_df["_task_order"] = result_df["task"].map(task_order)
    result_df = result_df.sort_values(["_category_order", "format", "_task_order"]).drop(columns=["_category_order", "_task_order"])

    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(result_df.to_string(index=False))

    print("\n全部完成。")
    print(f"结果表 CSV: {RESULT_CSV}")
    print(f"结果表 TXT: {RESULT_TXT}")
    print(f"查询预览目录: {QUERY_OUTPUT_DIR}")

    print("\n结果预览：")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()