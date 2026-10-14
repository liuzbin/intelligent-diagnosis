"""
评测指标。分三层：检索层、重排层、端到端层。

为什么要分层：端到端 Top3 命中率掉了，必须能定位是"没召回到"（检索问题）、
"召回到了但排在后面"（重排问题）、还是"排在前面但 LLM 没用上"（prompt 问题）。
这个分层定位方法在阶段 6 的 Bad case 分析中会反复用到。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .schema import DiagnosisResult, RetrievedItem


# ----------------------------------------------------------------------------
# 检索层 / 重排层
# ----------------------------------------------------------------------------

def _is_hit(item: RetrievedItem, gold: set[str]) -> bool:
    # 命中判定：候选工单的"标准根因类"落在可接受根因集合内
    return item.root_cause_id is not None and item.root_cause_id in gold


def recall_at_k(items: Sequence[RetrievedItem], gold: set[str], k: int) -> float:
    """Recall@K：前 K 个候选中是否出现过正确根因类（单样本 0/1，外层求平均）。

    注意：这里的 recall 是"根因类级别"的命中，不是传统 IR 的"相关文档覆盖率"。
    因为下游只关心能不能把正确根因交给 LLM，而不是找全所有相关工单。
    """
    return 1.0 if any(_is_hit(x, gold) for x in items[:k]) else 0.0


def mrr_at_k(items: Sequence[RetrievedItem], gold: set[str], k: int = 10) -> float:
    for rank, x in enumerate(items[:k], start=1):
        if _is_hit(x, gold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(items: Sequence[RetrievedItem], gold: set[str], k: int = 10) -> float:
    """二元相关性的 NDCG：重排阶段的主指标。"""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, x in enumerate(items[:k], start=1)
        if _is_hit(x, gold)
    )
    # 理想情况：前 n 个全是正确的（n = 实际命中个数，至少 1）
    n_rel = max(1, sum(1 for x in items[:k] if _is_hit(x, gold)))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_rel + 1))
    return dcg / idcg


# ----------------------------------------------------------------------------
# 端到端层
# ----------------------------------------------------------------------------

def top_k_hit(results: Sequence[DiagnosisResult], gold: set[str], k: int = 3) -> float:
    """项目主指标：Top3 命中率（70% → 85% 即此口径）。"""
    return 1.0 if any(r.root_cause_id in gold for r in results[:k]) else 0.0


def expected_calibration_error(
    pairs: Sequence[tuple[float, bool]], n_bins: int = 10
) -> float:
    """置信度校准误差 ECE。

    pairs: (模型给出的置信度, 该条结论是否正确)
    把置信度分桶，比较每个桶的"平均置信度"与"实际正确率"。
    业务含义：工程师看到 0.8 的置信度，应该真的有大约 80% 的概率是对的，
    否则置信度就是噪音（阶段 4 设计置信度时的验收指标）。
    """
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, ok in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, ok))
    total = len(pairs) or 1
    ece = 0.0
    for bucket in bins.values():
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += len(bucket) / total * abs(avg_conf - acc)
    return ece


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """命中率的 95% 置信区间（Wilson 区间）。

    用途：判断两个版本的差距是不是"真提升"。1000 条样本、命中率 70% 时，
    区间半宽约 ±2.8%，所以小于 3 个点的差异不能下结论。
    """
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half
