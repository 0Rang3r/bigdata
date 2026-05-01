import gc
import json
import math
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

import orjson
import pandas as pd
import ujson

INPUT_FILE = Path("data_stage/normalized.jsonl")
OUTPUT_CSV = Path("benchmarks") / "library_compare_results.csv"
OUTPUT_TXT = Path("benchmarks") / "library_compare_results.txt"

REPEAT = 3
SAMPLE_ROWS = 2000

# 尝试加载 C++ 绑定模块（第一次运行前需要先 build）
try:
    import cpp_jsonl_fastscan  # type: ignore
    CPP_BINDING_AVAILABLE = True
except ImportError:
    cpp_jsonl_fastscan = None
    CPP_BINDING_AVAILABLE = False


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


def load_sample_records(n=2000):
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


def load_sample_lines(n=2000):
    lines = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line)
    return lines


def process_lines_chunk(lines):
    total = 0
    matched = 0
    sum_text_len = 0

    for line in lines:
        obj = json.loads(line)
        total += 1
        if obj.get("domain", "") != "" and obj.get("text_len", 0) > 1000:
            matched += 1
            sum_text_len += obj.get("text_len", 0)

    return total, matched, sum_text_len


def scan_raw_python():
    total = 0
    matched = 0
    sum_text_len = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            if obj.get("domain", "") != "" and obj.get("text_len", 0) > 1000:
                matched += 1
                sum_text_len += obj.get("text_len", 0)

    avg_text_len = sum_text_len / matched if matched else 0.0
    return {"rows": total, "matched_rows": matched, "avg_text_len": avg_text_len}


def scan_raw_python_mp():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    workers = min(4, os.cpu_count() or 1)
    chunk_size = math.ceil(len(lines) / workers)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with mp.Pool(workers) as pool:
        stats = pool.map(process_lines_chunk, chunks)

    total = sum(x[0] for x in stats)
    matched = sum(x[1] for x in stats)
    sum_text_len = sum(x[2] for x in stats)

    avg_text_len = sum_text_len / matched if matched else 0.0
    return {"rows": total, "matched_rows": matched, "avg_text_len": avg_text_len}


def scan_cpp_binding():
    if not CPP_BINDING_AVAILABLE:
        raise RuntimeError(
            "C++ binding module cpp_jsonl_fastscan is not available. "
            "Please build it first: python setup_cpp_binding.py build_ext --inplace"
        )
    return cpp_jsonl_fastscan.scan_jsonl_stats(str(INPUT_FILE), threshold=1000)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"没找到输入文件: {INPUT_FILE}")

    rows = []

    sample_records = load_sample_records(SAMPLE_ROWS)
    sample_lines = load_sample_lines(SAMPLE_ROWS)

    # ---------- dumps ----------
    def stdlib_dump():
        s = json.dumps(sample_records, ensure_ascii=False)
        return len(s)

    def orjson_dump():
        b = orjson.dumps(sample_records)
        return len(b)

    def ujson_dump():
        s = ujson.dumps(sample_records, ensure_ascii=False)
        return len(s)

    # ---------- loads ----------
    stdlib_payload = json.dumps(sample_records, ensure_ascii=False)
    orjson_payload = orjson.dumps(sample_records)
    ujson_payload = ujson.dumps(sample_records, ensure_ascii=False)

    def stdlib_load():
        x = json.loads(stdlib_payload)
        return len(x)

    def orjson_load():
        x = orjson.loads(orjson_payload)
        return len(x)

    def ujson_load():
        x = ujson.loads(ujson_payload)
        return len(x)

    # ---------- parse NDJSON sample ----------
    def stdlib_parse_lines():
        cnt = 0
        for line in sample_lines:
            _ = json.loads(line)
            cnt += 1
        return cnt

    def orjson_parse_lines():
        cnt = 0
        for line in sample_lines:
            _ = orjson.loads(line.encode("utf-8"))
            cnt += 1
        return cnt

    def ujson_parse_lines():
        cnt = 0
        for line in sample_lines:
            _ = ujson.loads(line)
            cnt += 1
        return cnt

    tasks = [
        ("json", "stdlib", "dumps_array", stdlib_dump),
        ("json", "orjson", "dumps_array", orjson_dump),
        ("json", "ujson", "dumps_array", ujson_dump),
        ("json", "stdlib", "loads_array", stdlib_load),
        ("json", "orjson", "loads_array", orjson_load),
        ("json", "ujson", "loads_array", ujson_load),
        ("jsonl", "stdlib", "parse_sample_lines", stdlib_parse_lines),
        ("jsonl", "orjson", "parse_sample_lines", orjson_parse_lines),
        ("jsonl", "ujson", "parse_sample_lines", ujson_parse_lines),
        ("rawpython", "stdlib_json", "full_scan_single_process", scan_raw_python),
        ("rawpython", "multiprocessing", "full_scan_multi_process", scan_raw_python_mp),
    ]

    if CPP_BINDING_AVAILABLE:
        tasks.append(("rawpython", "cpp_binding", "full_scan_cpp_binding", scan_cpp_binding))

    print("开始库对比 benchmark ...")
    if CPP_BINDING_AVAILABLE:
        print("检测到 C++ 绑定模块：cpp_jsonl_fastscan")
    else:
        print("未检测到 C++ 绑定模块，当前仅运行 Python 版本 benchmark。")
        print("如果要启用 C++ 绑定，请先执行：python setup_cpp_binding.py build_ext --inplace")

    for category, engine, task, func in tasks:
        print(f"  -> {category} / {engine} / {task}")
        r = benchmark(func, repeat=REPEAT)
        rows.append({
            "category": category,
            "engine": engine,
            "task": task,
            "cpu_time_avg_sec": r["cpu_time_avg_sec"],
            "wall_time_avg_sec": r["wall_time_avg_sec"],
            "cpu_time_min_sec": r["cpu_time_min_sec"],
            "wall_time_min_sec": r["wall_time_min_sec"],
            "cpu_time_max_sec": r["cpu_time_max_sec"],
            "wall_time_max_sec": r["wall_time_max_sec"],
            "result_info": str(r["last_result"]),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(result_df.to_string(index=False))

    print("
全部完成。")
    print(f"结果表 CSV: {OUTPUT_CSV}")
    print(f"结果表 TXT: {OUTPUT_TXT}")
    print("
结果预览：")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
