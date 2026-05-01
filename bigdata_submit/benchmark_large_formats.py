import gc
import time
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.orc as orc

# =========================
# 路径设置（大样本）
# =========================
BASE_DIR = Path(".")

JSONL_FILE = BASE_DIR / "data_stage_large" / "normalized.jsonl"

JSON_FILE = BASE_DIR / "data_formats_large" / "json" / "docs.json"
CSV_FILE = BASE_DIR / "data_formats_large" / "csv" / "docs.csv"
TSV_FILE = BASE_DIR / "data_formats_large" / "tsv" / "docs.tsv"
PARQUET_FILE = BASE_DIR / "data_formats_large" / "parquet" / "docs.parquet"
ORC_DIR = BASE_DIR / "data_formats_large" / "orc_parts"
SQLITE_FILE = BASE_DIR / "data_formats_large" / "sqlite" / "docs.db"

OUTPUT_DIR = BASE_DIR / "benchmarks_large"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "benchmark_large_results.csv"
RESULT_TXT = OUTPUT_DIR / "benchmark_large_results.txt"

SQLITE_TABLE = "docs"

# 大样本默认先跑 1 次；你机器扛得住再改成 2
REPEAT = 1

# 是否对 JSON / JSONL 也做 DuckDB 查询
RUN_JSON_QUERIES = True
RUN_JSONL_QUERIES = True

# ORC 这里先做 file_size + read benchmark，不做 SQL 查询
RUN_ORC_QUERY = False

# DuckDB 读取大文本 CSV/TSV 时，单条记录可能非常长。
# 原脚本这里没有设置上限，容易触发 CSV Error on Line / max_line_size 报错。
# 先给一个较大的 64MB 上限；如果后面还有同类报错，再改到 128MB 或 256MB。
CSV_TSV_MAX_LINE_SIZE = 64 * 1024 * 1024
CSV_TSV_SAMPLE_SIZE = 20480


# =========================
# 工具函数
# =========================
def sizeof_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def ensure_inputs():
    required = [JSONL_FILE, JSON_FILE, CSV_FILE, TSV_FILE, PARQUET_FILE, SQLITE_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))
    if not ORC_DIR.exists():
        raise FileNotFoundError(f"缺少 ORC 目录：{ORC_DIR}")


def benchmark(func, repeat=1):
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


# =========================
# DuckDB 数据源辅助
# =========================
def _escape(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def duck_from_csv():
    return (
        f"read_csv_auto("
        f"'{_escape(CSV_FILE)}', "
        f"header=true, "
        f"sample_size={CSV_TSV_SAMPLE_SIZE}, "
        f"max_line_size={CSV_TSV_MAX_LINE_SIZE}"
        f")"
    )


def duck_from_tsv():
    return (
        f"read_csv_auto("
        f"'{_escape(TSV_FILE)}', "
        f"header=true, "
        f"delim='\\t', "
        f"sample_size={CSV_TSV_SAMPLE_SIZE}, "
        f"max_line_size={CSV_TSV_MAX_LINE_SIZE}"
        f")"
    )


def duck_from_parquet():
    return f"read_parquet('{_escape(PARQUET_FILE)}')"


def duck_from_json():
    # 大 JSON 数组
    return f"read_json_auto('{_escape(JSON_FILE)}')"


def duck_from_jsonl():
    # 优先用 ndjson；如果版本不支持，再回退 json_auto
    return None


def run_duck_sql_from_clause(from_clause: str, sql_template: str):
    con = duckdb.connect(database=":memory:")
    try:
        sql = sql_template.format(from_clause=from_clause)
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def run_duck_jsonl_sql(sql_template: str):
    con = duckdb.connect(database=":memory:")
    try:
        try:
            sql = sql_template.format(from_clause=f"read_ndjson_auto('{_escape(JSONL_FILE)}')")
            return con.execute(sql).fetchdf()
        except Exception:
            sql = sql_template.format(from_clause=f"read_json_auto('{_escape(JSONL_FILE)}')")
            return con.execute(sql).fetchdf()
    finally:
        con.close()


# =========================
# SQLite 辅助
# =========================
def run_sqlite_sql(sql: str):
    with sqlite3.connect(SQLITE_FILE) as conn:
        return pd.read_sql_query(sql, conn)


# =========================
# ORC 读取 benchmark
# =========================
def read_orc_row_count():
    total_rows = 0
    total_files = 0

    for p in sorted(ORC_DIR.glob("*.orc")):
        of = orc.ORCFile(p)
        total_rows += of.nrows
        total_files += 1

    return {"orc_files": total_files, "rows": total_rows}


# =========================
# 读取 benchmark（大样本）
# 这里用“扫描并计数”代替“整表读进 Python 内存”
# =========================
READ_COUNT_SQL = """
SELECT COUNT(*) AS n_rows
FROM {from_clause};
"""


def read_csv_count():
    return run_duck_sql_from_clause(duck_from_csv(), READ_COUNT_SQL).to_dict(orient="records")[0]


def read_tsv_count():
    return run_duck_sql_from_clause(duck_from_tsv(), READ_COUNT_SQL).to_dict(orient="records")[0]


def read_parquet_count():
    return run_duck_sql_from_clause(duck_from_parquet(), READ_COUNT_SQL).to_dict(orient="records")[0]


def read_json_count():
    return run_duck_sql_from_clause(duck_from_json(), READ_COUNT_SQL).to_dict(orient="records")[0]


def read_jsonl_count():
    return run_duck_jsonl_sql(READ_COUNT_SQL).to_dict(orient="records")[0]


def read_sqlite_count():
    with sqlite3.connect(SQLITE_FILE) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {SQLITE_TABLE}").fetchone()[0]
    return {"n_rows": row}


# =========================
# SQL-like 查询模板
# =========================
QUERY_FILTER = """
SELECT COUNT(*) AS matched_rows,
       AVG(text_len) AS avg_text_len
FROM {from_clause}
WHERE domain <> ''
  AND text_len > 1000;
"""

QUERY_GROUP = """
SELECT domain,
       COUNT(*) AS doc_cnt,
       AVG(text_len) AS avg_text_len
FROM {from_clause}
WHERE domain <> ''
GROUP BY domain
ORDER BY doc_cnt DESC, domain
LIMIT 20;
"""

QUERY_WINDOW = """
SELECT COUNT(*) AS n_rows,
       MAX(rn) AS max_rn
FROM (
    SELECT ROW_NUMBER() OVER (
               PARTITION BY domain
               ORDER BY warc_date
           ) AS rn
    FROM {from_clause}
    WHERE domain <> ''
) t;
"""


# =========================
# 主流程
# =========================
def main():
    ensure_inputs()

    rows = []

    # -------------------------
    # 文件大小
    # -------------------------
    rows.append({
        "category": "file_size",
        "format": "jsonl",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{sizeof_mb(JSONL_FILE):.2f} MB",
    })
    rows.append({
        "category": "file_size",
        "format": "json",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{sizeof_mb(JSON_FILE):.2f} MB",
    })
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
        "format": "tsv",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{sizeof_mb(TSV_FILE):.2f} MB",
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
        "format": "orc_dir",
        "engine": "filesystem",
        "task": "file_size_mb",
        "cpu_time_avg_sec": None,
        "wall_time_avg_sec": None,
        "cpu_time_min_sec": None,
        "wall_time_min_sec": None,
        "cpu_time_max_sec": None,
        "wall_time_max_sec": None,
        "result_info": f"{dir_size_mb(ORC_DIR):.2f} MB",
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
    print("1) 大样本读取 benchmark ...")

    tasks_read = [
        ("jsonl", "duckdb", read_jsonl_count),
        ("json", "duckdb", read_json_count),
        ("csv", "duckdb", read_csv_count),
        ("tsv", "duckdb", read_tsv_count),
        ("parquet", "duckdb", read_parquet_count),
        ("orc_dir", "pyarrow.orc", read_orc_row_count),
        ("sqlite", "sqlite3", read_sqlite_count),
    ]

    for fmt, engine, func in tasks_read:
        print(f"  -> read / {fmt}")
        r = benchmark(func, repeat=REPEAT)
        rows.append({
            "category": "read",
            "format": fmt,
            "engine": engine,
            "task": "count_scan",
            **{k: v for k, v in r.items() if k != "last_result"},
            "result_info": str(r["last_result"]),
        })

    # -------------------------
    # 查询 benchmark
    # -------------------------
    print("2) 大样本查询 benchmark ...")

    query_jobs = []

    if RUN_JSONL_QUERIES:
        query_jobs.extend([
            ("jsonl", "duckdb", "filter", lambda: run_duck_jsonl_sql(QUERY_FILTER)),
            ("jsonl", "duckdb", "group_by", lambda: run_duck_jsonl_sql(QUERY_GROUP)),
            ("jsonl", "duckdb", "window", lambda: run_duck_jsonl_sql(QUERY_WINDOW)),
        ])

    if RUN_JSON_QUERIES:
        query_jobs.extend([
            ("json", "duckdb", "filter", lambda: run_duck_sql_from_clause(duck_from_json(), QUERY_FILTER)),
            ("json", "duckdb", "group_by", lambda: run_duck_sql_from_clause(duck_from_json(), QUERY_GROUP)),
            ("json", "duckdb", "window", lambda: run_duck_sql_from_clause(duck_from_json(), QUERY_WINDOW)),
        ])

    query_jobs.extend([
        ("csv", "duckdb", "filter", lambda: run_duck_sql_from_clause(duck_from_csv(), QUERY_FILTER)),
        ("csv", "duckdb", "group_by", lambda: run_duck_sql_from_clause(duck_from_csv(), QUERY_GROUP)),
        ("csv", "duckdb", "window", lambda: run_duck_sql_from_clause(duck_from_csv(), QUERY_WINDOW)),

        ("tsv", "duckdb", "filter", lambda: run_duck_sql_from_clause(duck_from_tsv(), QUERY_FILTER)),
        ("tsv", "duckdb", "group_by", lambda: run_duck_sql_from_clause(duck_from_tsv(), QUERY_GROUP)),
        ("tsv", "duckdb", "window", lambda: run_duck_sql_from_clause(duck_from_tsv(), QUERY_WINDOW)),

        ("parquet", "duckdb", "filter", lambda: run_duck_sql_from_clause(duck_from_parquet(), QUERY_FILTER)),
        ("parquet", "duckdb", "group_by", lambda: run_duck_sql_from_clause(duck_from_parquet(), QUERY_GROUP)),
        ("parquet", "duckdb", "window", lambda: run_duck_sql_from_clause(duck_from_parquet(), QUERY_WINDOW)),

        ("sqlite", "sqlite3", "filter", lambda: run_sqlite_sql(QUERY_FILTER.format(from_clause=SQLITE_TABLE))),
        ("sqlite", "sqlite3", "group_by", lambda: run_sqlite_sql(QUERY_GROUP.format(from_clause=SQLITE_TABLE))),
        ("sqlite", "sqlite3", "window", lambda: run_sqlite_sql(QUERY_WINDOW.format(from_clause=SQLITE_TABLE))),
    ])

    for fmt, engine, task, func in query_jobs:
        print(f"  -> query / {fmt} / {task}")
        r = benchmark(func, repeat=REPEAT)

        if task == "group_by":
            info = f"rows={len(r['last_result'])}"
        else:
            info = str(r["last_result"].to_dict(orient="records")[0])

        rows.append({
            "category": "query",
            "format": fmt,
            "engine": engine,
            "task": task,
            **{k: v for k, v in r.items() if k != "last_result"},
            "result_info": info,
        })

    # -------------------------
    # 输出
    # -------------------------
    result_df = pd.DataFrame(rows)

    category_order = {"file_size": 0, "read": 1, "query": 2}
    task_order = {"file_size_mb": 0, "count_scan": 1, "filter": 2, "group_by": 3, "window": 4}

    result_df["_category_order"] = result_df["category"].map(category_order)
    result_df["_task_order"] = result_df["task"].map(task_order)
    result_df = result_df.sort_values(
        by=["_category_order", "format", "_task_order"],
        kind="stable"
    ).drop(columns=["_category_order", "_task_order"])

    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(result_df.to_string(index=False))

    print("\n全部完成。")
    print(f"结果表 CSV: {RESULT_CSV}")
    print(f"结果表 TXT: {RESULT_TXT}")
    print("\n结果预览：")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
