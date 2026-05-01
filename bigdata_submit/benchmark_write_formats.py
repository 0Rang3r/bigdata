import gc
import json
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.orc as orc
from sqlalchemy import create_engine

INPUT_FILE = Path("data_stage/normalized.jsonl")

OUTPUT_DIR = Path("benchmarks") / "write_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = Path("benchmarks") / "write_benchmark_results.csv"
RESULT_TXT = Path("benchmarks") / "write_benchmark_results.txt"

REPEAT = 3
TABLE_NAME = "docs"


def benchmark(func, repeat=3):
    cpu_times = []
    wall_times = []

    for _ in range(repeat):
        gc.collect()

        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        result = func()

        cpu_times.append(time.process_time() - t0_cpu)
        wall_times.append(time.perf_counter() - t0_wall)

    return {
        "cpu_time_avg_sec": sum(cpu_times) / len(cpu_times),
        "wall_time_avg_sec": sum(wall_times) / len(wall_times),
        "cpu_time_min_sec": min(cpu_times),
        "wall_time_min_sec": min(wall_times),
        "cpu_time_max_sec": max(cpu_times),
        "wall_time_max_sec": max(wall_times),
        "result": result,
    }


def sizeof_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def remove_if_exists(path: Path):
    if path.exists():
        path.unlink()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"没找到输入文件: {INPUT_FILE}")

    print("1) 读取 normalized.jsonl 到内存 ...")
    df = pl.read_ndjson(INPUT_FILE)

    wanted_cols = [
        "doc_id",
        "source_file",
        "url",
        "domain",
        "warc_date",
        "text",
        "text_len",
        "line_count",
    ]
    existing_cols = [c for c in wanted_cols if c in df.columns]
    df = df.select(existing_cols)

    pdf = df.to_pandas()
    records = pdf.to_dict(orient="records")

    rows = []

    json_file = OUTPUT_DIR / "docs.json"
    jsonl_file = OUTPUT_DIR / "docs.jsonl"
    csv_file = OUTPUT_DIR / "docs.csv"
    tsv_file = OUTPUT_DIR / "docs.tsv"
    parquet_file = OUTPUT_DIR / "docs.parquet"
    orc_file = OUTPUT_DIR / "docs.orc"
    sqlite_file = OUTPUT_DIR / "docs.db"

    def write_json():
        remove_if_exists(json_file)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        return sizeof_mb(json_file)

    def write_jsonl():
        remove_if_exists(jsonl_file)
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return sizeof_mb(jsonl_file)

    def write_csv():
        remove_if_exists(csv_file)
        df.write_csv(csv_file)
        return sizeof_mb(csv_file)

    def write_tsv():
        remove_if_exists(tsv_file)
        df.write_csv(tsv_file, separator="\t")
        return sizeof_mb(tsv_file)

    def write_parquet():
        remove_if_exists(parquet_file)
        df.write_parquet(parquet_file, compression="zstd")
        return sizeof_mb(parquet_file)

    def write_orc():
        remove_if_exists(orc_file)
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        orc.write_table(table, orc_file)
        return sizeof_mb(orc_file)

    def write_sqlite():
        remove_if_exists(sqlite_file)
        engine = create_engine(f"sqlite:///{sqlite_file}")
        pdf.to_sql(TABLE_NAME, engine, if_exists="replace", index=False, chunksize=2000)
        return sizeof_mb(sqlite_file)

    tasks = [
        ("json", "stdlib_json", write_json),
        ("jsonl", "stdlib_json", write_jsonl),
        ("csv", "polars", write_csv),
        ("tsv", "polars", write_tsv),
        ("parquet", "polars", write_parquet),
        ("orc", "pyarrow", write_orc),
        ("sqlite", "pandas+sqlalchemy", write_sqlite),
    ]

    print("2) 开始写入 benchmark ...")
    for fmt, engine, func in tasks:
        print(f"  -> {fmt}")
        r = benchmark(func, repeat=REPEAT)
        rows.append({
            "format": fmt,
            "engine": engine,
            "task": "write",
            "cpu_time_avg_sec": r["cpu_time_avg_sec"],
            "wall_time_avg_sec": r["wall_time_avg_sec"],
            "cpu_time_min_sec": r["cpu_time_min_sec"],
            "wall_time_min_sec": r["wall_time_min_sec"],
            "cpu_time_max_sec": r["cpu_time_max_sec"],
            "wall_time_max_sec": r["wall_time_max_sec"],
            "file_size_mb": round(r["result"], 2),
        })

    result_df = pd.DataFrame(rows)
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