import os
import csv
import gzip
import json
import shutil
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.orc as orc

# ========= 配置 =========
WET_DIR = Path("data_raw/wet_full")
TARGET_SOURCE_GB = 10.5       # 选取本地 WET 文件，累计压缩大小达到这个值
BATCH_ROWS = 5000             # 每批写入 Parquet / ORC / SQLite 的记录数

STAGE_DIR = Path("data_stage_large")
FORMATS_DIR = Path("data_formats_large")

JSONL_FILE = STAGE_DIR / "normalized.jsonl"

JSON_FILE = FORMATS_DIR / "json" / "docs.json"
CSV_FILE = FORMATS_DIR / "csv" / "docs.csv"
TSV_FILE = FORMATS_DIR / "tsv" / "docs.tsv"
PARQUET_FILE = FORMATS_DIR / "parquet" / "docs.parquet"
ORC_PARTS_DIR = FORMATS_DIR / "orc_parts"
SQLITE_FILE = FORMATS_DIR / "sqlite" / "docs.db"
MANIFEST_FILE = STAGE_DIR / "selected_wet_manifest.csv"
# =======================

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

def human_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GB"

def safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1

def decode_payload(payload: bytes) -> str:
    encodings = ["utf-8", "gb18030", "gbk", "big5"]
    for enc in encodings:
        try:
            return payload.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace").strip()

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()

def iter_warc_records(gz_path: Path):
    with gzip.open(gz_path, "rb") as f:
        pending_line = None

        while True:
            if pending_line is None:
                line = f.readline()
            else:
                line = pending_line
                pending_line = None

            if not line:
                break

            if not line.startswith(b"WARC/"):
                continue

            headers = {}

            while True:
                h = f.readline()
                if not h:
                    return
                if h in (b"\n", b"\r\n"):
                    break

                if b":" in h:
                    key, value = h.decode("utf-8", errors="replace").split(":", 1)
                    headers[key.strip()] = value.strip()

            content_length = int(headers.get("Content-Length", "0") or 0)
            payload = f.read(content_length) if content_length > 0 else b""

            while True:
                nxt = f.readline()
                if not nxt:
                    break
                if nxt.startswith(b"WARC/"):
                    pending_line = nxt
                    break
                if nxt in (b"\n", b"\r\n"):
                    continue

            yield headers, payload

def select_source_files():
    files = sorted(WET_DIR.glob("*.warc.wet.gz"))
    if not files:
        raise FileNotFoundError(f"在 {WET_DIR} 里没找到任何 .warc.wet.gz 文件")

    target_bytes = int(TARGET_SOURCE_GB * (1024 ** 3))
    selected = []
    total = 0

    for f in files:
        selected.append(f)
        total += f.stat().st_size
        if total >= target_bytes:
            break

    if total < target_bytes:
        raise RuntimeError(
            f"本地 WET 文件总量不足。当前只有 {human_gb(total)}，目标是 {human_gb(target_bytes)}。"
        )

    return selected, total

def prepare_output_dirs():
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (FORMATS_DIR / "json").mkdir(parents=True, exist_ok=True)
    (FORMATS_DIR / "csv").mkdir(parents=True, exist_ok=True)
    (FORMATS_DIR / "tsv").mkdir(parents=True, exist_ok=True)
    (FORMATS_DIR / "parquet").mkdir(parents=True, exist_ok=True)
    (FORMATS_DIR / "sqlite").mkdir(parents=True, exist_ok=True)

    if ORC_PARTS_DIR.exists():
        shutil.rmtree(ORC_PARTS_DIR)
    ORC_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    for p in [JSONL_FILE, JSON_FILE, CSV_FILE, TSV_FILE, PARQUET_FILE, SQLITE_FILE, MANIFEST_FILE]:
        if p.exists():
            p.unlink()

def main():
    prepare_output_dirs()

    selected_files, selected_total_bytes = select_source_files()
    print(f"选中的 WET 文件数: {len(selected_files)}")
    print(f"选中源数据累计大小: {human_gb(selected_total_bytes)}")

    with open(MANIFEST_FILE, "w", newline="", encoding="utf-8-sig") as mf:
        mw = csv.writer(mf)
        mw.writerow(["source_file", "size_mb"])
        for f in selected_files:
            mw.writerow([f.name, round(f.stat().st_size / (1024 ** 2), 2)])

    conn = sqlite3.connect(SQLITE_FILE)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS docs")
    cur.execute("""
        CREATE TABLE docs (
            doc_id TEXT,
            source_file TEXT,
            url TEXT,
            domain TEXT,
            warc_date TEXT,
            text TEXT,
            text_len INTEGER,
            line_count INTEGER
        )
    """)
    conn.commit()

    parquet_writer = None
    orc_part_idx = 0
    batch_records = []
    sqlite_rows = []

    total_raw_records = 0
    total_written_rows = 0

    with open(JSONL_FILE, "w", encoding="utf-8") as jsonl_f, \
         open(JSON_FILE, "w", encoding="utf-8") as json_f, \
         open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as csv_f, \
         open(TSV_FILE, "w", newline="", encoding="utf-8-sig") as tsv_f:

        fieldnames = [
            "doc_id",
            "source_file",
            "url",
            "domain",
            "warc_date",
            "text",
            "text_len",
            "line_count",
        ]

        csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        tsv_writer = csv.DictWriter(tsv_f, fieldnames=fieldnames, delimiter="\t")

        csv_writer.writeheader()
        tsv_writer.writeheader()

        json_f.write("[\n")
        first_json_item = True

        def flush_batch():
            nonlocal parquet_writer, orc_part_idx, batch_records, sqlite_rows

            if not batch_records:
                return

            table = pa.Table.from_pylist(batch_records)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    PARQUET_FILE,
                    table.schema,
                    compression="zstd"
                )

            parquet_writer.write_table(table)

            orc_path = ORC_PARTS_DIR / f"part-{orc_part_idx:06d}.orc"
            orc.write_table(table, orc_path)
            orc_part_idx += 1

            cur.executemany("""
                INSERT INTO docs (
                    doc_id, source_file, url, domain, warc_date, text, text_len, line_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, sqlite_rows)
            conn.commit()

            batch_records = []
            sqlite_rows = []

        for file_idx, gz_path in enumerate(selected_files, 1):
            print(f"\n处理文件 [{file_idx}/{len(selected_files)}]: {gz_path.name}")

            file_raw_records = 0
            file_written_rows = 0

            try:
                for rec_idx, (headers, payload) in enumerate(iter_warc_records(gz_path), 1):
                    file_raw_records += 1
                    total_raw_records += 1

                    if headers.get("WARC-Type") != "conversion":
                        continue

                    text = clean_text(decode_payload(payload))
                    if not text:
                        continue

                    url = headers.get("WARC-Target-URI", "")
                    warc_date = headers.get("WARC-Date", "")
                    domain = safe_domain(url)

                    obj = {
                        "doc_id": f"{gz_path.stem}_{rec_idx}",
                        "source_file": gz_path.name,
                        "url": url,
                        "domain": domain,
                        "warc_date": warc_date,
                        "text": text,
                        "text_len": len(text),
                        "line_count": line_count(text),
                    }

                    line = json.dumps(obj, ensure_ascii=False)

                    jsonl_f.write(line + "\n")

                    if not first_json_item:
                        json_f.write(",\n")
                    json_f.write(line)
                    first_json_item = False

                    csv_writer.writerow(obj)
                    tsv_writer.writerow(obj)

                    batch_records.append(obj)
                    sqlite_rows.append((
                        obj["doc_id"],
                        obj["source_file"],
                        obj["url"],
                        obj["domain"],
                        obj["warc_date"],
                        obj["text"],
                        obj["text_len"],
                        obj["line_count"],
                    ))

                    file_written_rows += 1
                    total_written_rows += 1

                    if len(batch_records) >= BATCH_ROWS:
                        flush_batch()

                flush_batch()

            except EOFError:
                print(f"  警告：{gz_path.name} 是损坏或未完整下载的 gzip 文件，已跳过未完成部分。")
                flush_batch()

            print(f"  原始记录数: {file_raw_records}")
            print(f"  写入记录数: {file_written_rows}")

        json_f.write("\n]\n")

    if parquet_writer is not None:
        parquet_writer.close()

    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_domain ON docs(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_warc_date ON docs(warc_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_text_len ON docs(text_len)")
    conn.commit()

    row_count = cur.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    conn.close()

    print("\n全部完成。")
    print(f"选中源数据大小: {human_gb(selected_total_bytes)}")
    print(f"总原始记录数: {total_raw_records}")
    print(f"总写入行数: {total_written_rows}")
    print(f"SQLite 行数: {row_count}")

    print("\n输出大小：")
    print(f"JSONL   : {sizeof_mb(JSONL_FILE):.2f} MB")
    print(f"JSON    : {sizeof_mb(JSON_FILE):.2f} MB")
    print(f"CSV     : {sizeof_mb(CSV_FILE):.2f} MB")
    print(f"TSV     : {sizeof_mb(TSV_FILE):.2f} MB")
    print(f"Parquet : {sizeof_mb(PARQUET_FILE):.2f} MB")
    print(f"ORC目录 : {dir_size_mb(ORC_PARTS_DIR):.2f} MB")
    print(f"SQLite  : {sizeof_mb(SQLITE_FILE):.2f} MB")

    print("\n输出路径：")
    print(f"JSONL   -> {JSONL_FILE}")
    print(f"JSON    -> {JSON_FILE}")
    print(f"CSV     -> {CSV_FILE}")
    print(f"TSV     -> {TSV_FILE}")
    print(f"Parquet -> {PARQUET_FILE}")
    print(f"ORC dir -> {ORC_PARTS_DIR}")
    print(f"SQLite  -> {SQLITE_FILE}")
    print(f"Manifest-> {MANIFEST_FILE}")

if __name__ == "__main__":
    main()