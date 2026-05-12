"""
Layer 1：文档切片质量检查（共 5 项）
  1. 可视化所有 chunk（内容 + 字符数）
  2. chunk_size=500 合理性：接近上限的 chunk 数量
  3. chunk_overlap=50 充分性：相邻同源 chunk 是否有重叠
  4. 长度分布：极短（<100字）/ 极长（>480字）chunk 统计
  5. 表格完整性：表头和数据行是否在同一 chunk

运行方式（在 cycling-coach/ 目录下）：
    python eval/layer1_chunks.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
sys.path.insert(0, str(ROOT))


def _classify_table(content: str) -> str:
    """判断 chunk 中的表格状态。"""
    has_separator = "|---" in content or "| ---" in content
    pipe_count = content.count("|")
    has_data_row = pipe_count >= 6  # 至少两行含管道符

    if has_separator and has_data_row:
        return "intact"
    if has_separator and not has_data_row:
        return "header_only"   # 表头行被单独切出
    if not has_separator and pipe_count >= 4:
        return "data_only"     # 数据行没有表头
    return "no_table"


def _check_overlap(cur_content: str, nxt_content: str, overlap_setting: int) -> dict:
    """粗略检查相邻 chunk 的重叠：取当前 chunk 末尾 overlap_setting 字符，
    看是否出现在下一个 chunk 开头 overlap_setting*2 字符内。"""
    tail = cur_content[-overlap_setting:].strip()
    head = nxt_content[: overlap_setting * 2]
    # 取 tail 前 15 字做子串匹配（避免空白差异导致误判）
    probe = tail[:15]
    return {
        "overlap_found": len(probe) > 5 and probe in head,
        "tail_sample": tail[:30],
        "head_sample": head[:30],
    }


def run() -> dict:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import MarkdownTextSplitter

    # ── 加载文档（与 rag.py 保持一致，只索引 knowledge/）────────
    loader = DirectoryLoader(
        str(ROOT / "data/knowledge"),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # ── 分析每个 chunk ────────────────────────────────────────────
    records = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        source = Path(chunk.metadata.get("source", "unknown")).name
        records.append({
            "id": i,
            "source": source,
            "char_count": len(content),
            "table_status": _classify_table(content),
            "content": content,
        })

    char_counts = [r["char_count"] for r in records]
    short = [r for r in records if r["char_count"] < 100]
    near_limit = [r for r in records if r["char_count"] > 480]

    # ── overlap 检查（仅同一 source 的相邻 chunk）────────────────
    overlap_results = []
    for i in range(len(records) - 1):
        if records[i]["source"] != records[i + 1]["source"]:
            continue
        check = _check_overlap(records[i]["content"], records[i + 1]["content"], 50)
        overlap_results.append({
            "chunk_ids": [i, i + 1],
            "source": records[i]["source"],
            **check,
        })

    overlaps_found = sum(1 for o in overlap_results if o["overlap_found"])

    # ── 表格完整性 ────────────────────────────────────────────────
    table_chunks = [r for r in records if r["table_status"] != "no_table"]
    broken_tables = [r for r in table_chunks if r["table_status"] in ("header_only", "data_only")]

    # ── 构造 items ────────────────────────────────────────────────
    items = [
        {
            "id": "1",
            "name": "chunk_visualization",
            "status": "pass",
            "detail": {
                "total_docs": len(docs),
                "total_chunks": len(chunks),
                "sources": sorted({r["source"] for r in records}),
            },
        },
        {
            "id": "2",
            "name": "chunk_size_appropriateness",
            "status": "warn" if near_limit else "pass",
            "detail": {
                "chunk_size_setting": 500,
                "chunks_over_480": len(near_limit),
                "over_480_ids": [r["id"] for r in near_limit],
                "note": "超过480字的chunk存在在语义中间被截断的风险",
            },
        },
        {
            "id": "3",
            "name": "chunk_overlap_check",
            "status": "pass" if overlaps_found > 0 else "warn",
            "detail": {
                "chunk_overlap_setting": 50,
                "adjacent_pairs_checked": len(overlap_results),
                "pairs_with_overlap_found": overlaps_found,
                "samples": overlap_results[:5],
                "note": "overlap_found=0 说明切分点刚好不在overlap范围内，不一定是问题",
            },
        },
        {
            "id": "4",
            "name": "length_distribution",
            "status": "warn" if short else "pass",
            "detail": {
                "min_chars": min(char_counts),
                "max_chars": max(char_counts),
                "avg_chars": round(sum(char_counts) / len(char_counts), 1),
                "distribution": {
                    "<100": sum(1 for c in char_counts if c < 100),
                    "100-299": sum(1 for c in char_counts if 100 <= c < 300),
                    "300-499": sum(1 for c in char_counts if 300 <= c < 500),
                    ">=500": sum(1 for c in char_counts if c >= 500),
                },
                "short_chunk_ids": [r["id"] for r in short],
            },
        },
        {
            "id": "5",
            "name": "table_integrity",
            "status": "fail" if broken_tables else "pass",
            "detail": {
                "intact": sum(1 for r in table_chunks if r["table_status"] == "intact"),
                "header_only": sum(1 for r in table_chunks if r["table_status"] == "header_only"),
                "data_only": sum(1 for r in table_chunks if r["table_status"] == "data_only"),
                "broken_chunk_ids": [r["id"] for r in broken_tables],
            },
        },
    ]

    passed_count = sum(1 for i in items if i["status"] == "pass")
    result = {
        "layer": 1,
        "name": "chunks",
        "passed": not any(i["status"] == "fail" for i in items),
        "score": round(passed_count / len(items), 2),
        "items": items,
        "errors": [],
        "duration_s": None,
        "_chunk_contents": [
            {
                "id": r["id"],
                "source": r["source"],
                "char_count": r["char_count"],
                "table_status": r["table_status"],
                "content": r["content"],
            }
            for r in records
        ],
    }
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = run()
    result["duration_s"] = round(time.time() - t0, 2)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / f"layer1_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer1_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── 控制台摘要 ────────────────────────────────────────────────
    detail = result["items"]
    chunks_info = detail[0]["detail"]
    dist = detail[3]["detail"]["distribution"]

    print(f"\n=== Layer 1: 切片质量 ===")
    print(f"文档数：{chunks_info['total_docs']}  chunk总数：{chunks_info['total_chunks']}")
    print(f"来源文件：{', '.join(chunks_info['sources'])}")
    print(f"长度分布：<100={dist['<100']}  100-299={dist['100-299']}  "
          f"300-499={dist['300-499']}  >=500={dist['>=500']}")
    print()

    icons = {"pass": "✓", "warn": "△", "fail": "✗"}
    for item in result["items"]:
        icon = icons[item["status"]]
        print(f"  [{icon}] 项{item['id']} {item['name']}: {item['status']}")

    print(f"\n得分：{result['score']}  passed={result['passed']}")
    print(f"结果写入：{out_path}  耗时：{result['duration_s']}s")

    # ── 打印每个 chunk 供人工 review ──────────────────────────────
    print(f"\n{'─'*60}")
    print("各 chunk 内容预览（前80字）：")
    for c in result["_chunk_contents"]:
        preview = c["content"][:80].replace("\n", " ")
        table_flag = f" [表格:{c['table_status']}]" if c["table_status"] != "no_table" else ""
        print(f"  #{c['id']:02d} {c['source']} ({c['char_count']}字){table_flag}")
        print(f"      {preview}")
