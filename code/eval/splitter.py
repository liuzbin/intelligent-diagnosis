"""
黄金集构建：时间切分 + 分层抽样 + 近重复剔除。

核心原则：模拟"用历史预测未来"。
- 知识库：只包含 cutoff_kb 之前闭环的工单
- 评测集：取 [eval_start, eval_end) 闭环的工单
- 两者之间留空档，并对评测集做近重复剔除，防止同一故障被复制粘贴多次提单造成泄漏

如果不做这一步，POC 阶段曾出现过"评测集里的工单本身就在知识库里"，
命中率虚高 15 个点以上 —— 这是阶段 0 踩的第一个坑。
"""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime
from typing import Callable, Iterable

from .schema import GoldenSample, ProductLine


def time_split(
    samples: Iterable[GoldenSample],
    kb_cutoff: datetime,     # 例：2024-06-30
    eval_start: datetime,    # 例：2024-07-01
    eval_end: datetime,      # 例：2024-12-31
) -> list[GoldenSample]:
    assert kb_cutoff <= eval_start, "知识库截止时间必须早于评测集起始时间"
    return [s for s in samples if eval_start <= s.closed_at < eval_end]


def stratified_sample(
    samples: list[GoldenSample],
    total: int = 1000,
    min_per_line: int = 100,
    seed: int = 42,
) -> list[GoldenSample]:
    """按产品线分层抽样：每条产品线至少 min_per_line 条，余量按原始占比分配。"""
    rng = random.Random(seed)  # 固定种子，保证黄金集可复现
    by_line: dict[ProductLine, list[GoldenSample]] = defaultdict(list)
    for s in samples:
        by_line[s.request.product_line].append(s)

    picked: list[GoldenSample] = []
    remaining_quota = total
    for line, pool in by_line.items():
        k = min(min_per_line, len(pool))
        picked.extend(rng.sample(pool, k))
        remaining_quota -= k

    # 余量按产品线原始规模占比分配
    picked_ids = {s.sample_id for s in picked}
    rest_pool = [s for s in samples if s.sample_id not in picked_ids]
    if remaining_quota > 0 and rest_pool:
        picked.extend(rng.sample(rest_pool, min(remaining_quota, len(rest_pool))))
    return picked


def dedup_against_kb(
    eval_samples: list[GoldenSample],
    is_near_duplicate: Callable[[GoldenSample], bool],
) -> list[GoldenSample]:
    """剔除与知识库近重复的样本。

    is_near_duplicate 的实现由离线链路提供（阶段 2 的 MinHash/SimHash 去重模块复用），
    这里只依赖抽象，评测模块不关心具体算法。
    """
    return [s for s in eval_samples if not is_near_duplicate(s)]
