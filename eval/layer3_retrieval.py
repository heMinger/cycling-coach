"""
Layer 3：检索质量评估（共 6 项，对应清单 10-15）+ 阈值扫描
  10. Golden Query Set 构建（从 eval/data/golden_queries.json 加载）
  11. Context Precision：k=3 召回的 chunk 中有多少是真正相关的
  12. Context Recall：需要的 chunk 有没有被召回（按 expected_keywords 代理）
  13. 边界 query 测试：out_of_scope query 召回的 chunk 相关性
  14. k=3 vs k=5 对比：recall 能否提升，precision 是否下降
  15. 相关性分数分布：观察 in_scope vs out_of_scope 的分数差距
  _threshold_sweep：k=5 下扫描 L2 阈值 [0.80, 0.85, 0.90, 0.95, ∞]，
                   输出 precision/recall/平均返回数/out_scope拒绝率，用于确定最优阈值

运行方式（在 cycling-coach/ 目录下）：
    python eval/layer3_retrieval.py

判断说明：
  - "相关"的定义：召回 chunk 的内容包含 expected_keywords 中至少一个关键词
  - out_of_scope query 的 precision 应趋近 0（召回内容无关）
  - 使用 vectorstore.similarity_search_with_score() 获取距离分数
    Chroma 返回的是 L2 距离，值越小越相似（0=完全相同）
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_PATH = EVAL_DIR / "data" / "golden_queries.json"
sys.path.insert(0, str(ROOT))


def _load_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=str(ROOT / "chroma_db"), embedding_function=embeddings)


def _is_relevant(chunk_content: str, expected_keywords: list[str]) -> bool:
    """粗略相关性判断：chunk 内容包含至少一个 expected_keyword。"""
    if not expected_keywords:
        return False
    content_lower = chunk_content.lower()
    return any(kw.lower() in content_lower for kw in expected_keywords)


def _retrieve(vs, query: str, k: int) -> list[dict]:
    results = vs.similarity_search_with_score(query, k=k)
    return [
        {
            "source": Path(doc.metadata.get("source", "")).name,
            "char_count": len(doc.page_content),
            "score": round(float(score), 4),   # L2 距离，越小越相似
            "content_preview": doc.page_content[:100].replace("\n", " "),
            "content": doc.page_content,
        }
        for doc, score in results
    ]


def _eval_query(vs, q: dict, k: int) -> dict:
    hits = _retrieve(vs, q["query"], k)
    category = q["category"]
    expected_kw = q.get("expected_keywords", [])
    expected_src = q.get("expected_sources", [])

    relevant_hits = [h for h in hits if _is_relevant(h["content"], expected_kw)]

    if category == "out_of_scope":
        # out_of_scope：召回内容不应包含任何 expected_keywords（期望全部不相关）
        precision = round(1 - len(relevant_hits) / len(hits), 4) if hits else 1.0
        recall = None   # out_of_scope 不计算 recall
        status = "pass" if len(relevant_hits) == 0 else "fail"
    else:
        precision = round(len(relevant_hits) / len(hits), 4) if hits else 0.0
        # recall 代理：expected_sources 中的文件是否至少有一个被召回
        recalled_sources = {h["source"] for h in hits}
        sources_recalled = sum(1 for s in expected_src if s in recalled_sources)
        recall = round(sources_recalled / len(expected_src), 4) if expected_src else None
        status = "pass" if precision >= 0.5 else "fail"

    return {
        "id": q["id"],
        "query": q["query"],
        "category": category,
        "k": k,
        "precision": precision,
        "recall": recall,
        "status": status,
        "hits": [
            {k2: v for k2, v in h.items() if k2 != "content"}
            for h in hits
        ],
    }


def _eval_with_threshold(vs, q: dict, k: int, threshold: float) -> dict:
    """_eval_query 的阈值过滤变体，镜像生产代码行为（无 fallback）。

    空结果（all filtered）的语义：
      - out_of_scope query → 完美拒绝，precision=1.0
      - in_scope query     → 漏召回，precision=0.0，recall=0.0
    """
    raw = vs.similarity_search_with_score(q["query"], k=k)
    filtered = [(doc, score) for doc, score in raw if score < threshold]

    hits = [
        {
            "source": Path(doc.metadata.get("source", "")).name,
            "score": round(float(score), 4),
            "content": doc.page_content,
        }
        for doc, score in filtered
    ]

    category = q["category"]
    expected_kw = q.get("expected_keywords", [])
    expected_src = q.get("expected_sources", [])
    relevant_hits = [h for h in hits if _is_relevant(h["content"], expected_kw)]

    if category == "out_of_scope":
        # 空结果 = 完美拒绝，视为 precision=1.0
        precision = 1.0 if not hits else round(1 - len(relevant_hits) / len(hits), 4)
        recall = None
    else:
        # 空结果 = 完全漏召回
        precision = round(len(relevant_hits) / len(hits), 4) if hits else 0.0
        recalled_sources = {h["source"] for h in hits}
        sources_recalled = sum(1 for s in expected_src if s in recalled_sources)
        recall = round(sources_recalled / len(expected_src), 4) if expected_src else None

    return {"precision": precision, "recall": recall, "hits_count": len(hits)}


def _threshold_sweep(vs, golden_queries: list) -> list[dict]:
    """扫描不同 L2 阈值，输出 precision/recall/平均召回数/out_scope 拒绝率。"""
    THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 9999]  # 9999 = 无过滤
    in_qs = [q for q in golden_queries if q["category"] in ("in_scope", "in_scope_multi")]
    out_qs = [q for q in golden_queries if q["category"] == "out_of_scope"]

    rows = []
    for t in THRESHOLDS:
        in_evals = [_eval_with_threshold(vs, q, k=5, threshold=t) for q in in_qs]
        out_evals = [_eval_with_threshold(vs, q, k=5, threshold=t) for q in out_qs]

        avg_precision = round(sum(e["precision"] for e in in_evals) / len(in_evals), 4)
        valid_recalls = [e["recall"] for e in in_evals if e["recall"] is not None]
        avg_recall = round(sum(valid_recalls) / len(valid_recalls), 4) if valid_recalls else 0.0
        avg_hits = round(sum(e["hits_count"] for e in in_evals) / len(in_evals), 2)
        rejection = round(sum(e["precision"] for e in out_evals) / len(out_evals), 4)

        rows.append({
            "threshold": "∞" if t == 9999 else t,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_hits_returned": avg_hits,
            "out_scope_rejection_rate": rejection,
        })
    return rows


def run() -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden_queries = json.load(f)

    vs = _load_vectorstore()

    # ── 项 10：Golden Query Set 覆盖统计 ─────────────────────────
    categories = {}
    for q in golden_queries:
        categories.setdefault(q["category"], []).append(q["id"])

    item10 = {
        "id": "10",
        "name": "golden_query_set",
        "status": "pass",
        "detail": {
            "total_queries": len(golden_queries),
            "by_category": {k: len(v) for k, v in categories.items()},
            "query_ids": [q["id"] for q in golden_queries],
        },
    }

    # ── 项 11+12+13：k=3 逐条评估 ────────────────────────────────
    results_k3 = [_eval_query(vs, q, k=3) for q in golden_queries]

    in_scope = [r for r in results_k3 if r["category"] in ("in_scope", "in_scope_multi")]
    out_scope = [r for r in results_k3 if r["category"] == "out_of_scope"]

    precision_k3 = round(
        sum(r["precision"] for r in in_scope) / len(in_scope), 4
    ) if in_scope else 0.0
    recall_k3 = round(
        sum(r["recall"] for r in in_scope if r["recall"] is not None)
        / sum(1 for r in in_scope if r["recall"] is not None),
        4,
    ) if in_scope else 0.0

    item11 = {
        "id": "11",
        "name": "context_precision_k3",
        "status": "pass" if precision_k3 >= 0.6 else "fail",
        "detail": {
            "avg_precision": precision_k3,
            "threshold": 0.6,
            "per_query": [
                {"id": r["id"], "query": r["query"], "precision": r["precision"], "status": r["status"]}
                for r in in_scope
            ],
        },
    }

    item12 = {
        "id": "12",
        "name": "context_recall_k3",
        "status": "pass" if recall_k3 >= 0.6 else "fail",
        "detail": {
            "avg_recall": recall_k3,
            "threshold": 0.6,
            "per_query": [
                {"id": r["id"], "query": r["query"], "recall": r["recall"], "status": r["status"]}
                for r in in_scope
            ],
        },
    }

    out_scope_pass = sum(1 for r in out_scope if r["status"] == "pass")
    item13 = {
        "id": "13",
        "name": "boundary_query_test",
        "status": "pass" if out_scope_pass == len(out_scope) else "warn",
        "detail": {
            "out_of_scope_queries": len(out_scope),
            "correctly_rejected": out_scope_pass,
            "per_query": [
                {
                    "id": r["id"],
                    "query": r["query"],
                    "status": r["status"],
                    "top_hit": r["hits"][0] if r["hits"] else None,
                }
                for r in out_scope
            ],
        },
    }

    # ── 项 14：k=3 vs k=5 对比 ───────────────────────────────────
    results_k5 = [_eval_query(vs, q, k=5) for q in golden_queries if q["category"] in ("in_scope", "in_scope_multi")]

    precision_k5 = round(sum(r["precision"] for r in results_k5) / len(results_k5), 4) if results_k5 else 0.0
    recall_k5 = round(
        sum(r["recall"] for r in results_k5 if r["recall"] is not None)
        / sum(1 for r in results_k5 if r["recall"] is not None),
        4,
    ) if results_k5 else 0.0

    item14 = {
        "id": "14",
        "name": "k3_vs_k5_comparison",
        "status": "pass",
        "detail": {
            "k3": {"avg_precision": precision_k3, "avg_recall": recall_k3},
            "k5": {"avg_precision": precision_k5, "avg_recall": recall_k5},
            "recall_gain": round(recall_k5 - recall_k3, 4),
            "precision_drop": round(precision_k3 - precision_k5, 4),
            "note": f"总chunk数=7，k=5时召回>70%的知识库，precision下降在预期内",
        },
    }

    # ── 项 15：相关性分数分布 ────────────────────────────────────
    in_scope_scores = [h["score"] for r in results_k3 if r["category"] in ("in_scope", "in_scope_multi") for h in r["hits"]]
    out_scope_scores = [h["score"] for r in results_k3 if r["category"] == "out_of_scope" for h in r["hits"]]

    def _stats(scores):
        if not scores:
            return {}
        return {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "avg": round(sum(scores) / len(scores), 4),
        }

    # L2 距离：in_scope 的分数应低于 out_of_scope（更相似）
    score_gap = round(
        (_stats(out_scope_scores).get("avg", 0) - _stats(in_scope_scores).get("avg", 0)),
        4,
    )
    item15 = {
        "id": "15",
        "name": "score_distribution",
        "status": "pass" if score_gap > 0 else "warn",
        "detail": {
            "note": "L2距离越小越相似；in_scope分数应低于out_of_scope",
            "in_scope_l2": _stats(in_scope_scores),
            "out_of_scope_l2": _stats(out_scope_scores),
            "gap_out_minus_in": score_gap,
        },
    }

    sweep = _threshold_sweep(vs, golden_queries)

    items = [item10, item11, item12, item13, item14, item15]
    passed_count = sum(1 for i in items if i["status"] == "pass")

    result = {
        "layer": 3,
        "name": "retrieval",
        "passed": not any(i["status"] == "fail" for i in items),
        "score": round(passed_count / len(items), 2),
        "items": items,
        "errors": [],
        "duration_s": None,
        "_per_query_detail": results_k3,
        "_threshold_sweep": sweep,
    }
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = run()
    result["duration_s"] = round(time.time() - t0, 2)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / f"layer3_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer3_retrieval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── 控制台摘要 ────────────────────────────────────────────────
    icons = {"pass": "✓", "warn": "△", "fail": "✗"}
    print(f"\n=== Layer 3: 检索质量 ===")
    for item in result["items"]:
        icon = icons[item["status"]]
        d = item["detail"]
        extra = ""
        if item["id"] == "11":
            extra = f"  precision={d['avg_precision']}"
        elif item["id"] == "12":
            extra = f"  recall={d['avg_recall']}"
        elif item["id"] == "13":
            extra = f"  correctly_rejected={d['correctly_rejected']}/{d['out_of_scope_queries']}"
        elif item["id"] == "14":
            extra = f"  k3={d['k3']}  k5={d['k5']}"
        elif item["id"] == "15":
            extra = f"  gap={d['gap_out_minus_in']} (out-in L2)"
        print(f"  [{icon}] 项{item['id']} {item['name']}: {item['status']}{extra}")

    print(f"\n得分：{result['score']}  passed={result['passed']}")
    print(f"结果写入：{out_path}  耗时：{result['duration_s']}s")

    # ── 打印阈值扫描表 ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("阈值扫描（k=5）：")
    print(f"  {'阈值':>6}  {'precision':>10}  {'recall':>8}  {'avg_hits':>9}  {'out拒绝率':>10}")
    for row in result["_threshold_sweep"]:
        t = row["threshold"]
        print(f"  {str(t):>6}  {row['avg_precision']:>10.4f}  {row['avg_recall']:>8.4f}"
              f"  {row['avg_hits_returned']:>9.2f}  {row['out_scope_rejection_rate']:>10.4f}")

    # ── 打印逐条 query 结果 ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print("逐条 query 结果（k=3）：")
    for r in result["_per_query_detail"]:
        icon = icons.get(r["status"], "?")
        p = f"precision={r['precision']}" if r["recall"] is not None else f"precision(rejection)={r['precision']}"
        rec = f" recall={r['recall']}" if r["recall"] is not None else ""
        print(f"  [{icon}] {r['id']} [{r['category']}] {p}{rec}")
        print(f"      Q: {r['query']}")
        for i, h in enumerate(r["hits"]):
            print(f"      #{i+1} {h['source']} (L2={h['score']}) {h['content_preview'][:60]}")
