"""
评测执行器：对一个"诊断链路版本"跑完整黄金集，输出分层指标。

用法（示意）：
    runner = EvalRunner(pipeline=online_pipeline, samples=golden_set)
    report = await runner.run(concurrency=8)

设计要点：
1. pipeline 是一个协议（Protocol），线上链路、POC 链路、消融实验链路都实现它，
   评测代码一行不改 —— 这样"换向量库/换 embedding/加重排"都能公平对比。
2. 中间结果（召回、重排）也要返回，用于分层定位问题。
3. 并发跑：1000 条 × 每条 3~5 秒，串行要 1 个多小时；用信号量控制并发，
   避免把共享的 LLM 推理服务打满（LLM 是平台算力组的共享资源）。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from . import metrics
from .schema import DiagnosisRequest, DiagnosisResult, GoldenSample, RetrievedItem


@dataclass
class PipelineTrace:
    """链路各阶段的中间产物。"""
    recalled: list[RetrievedItem] = field(default_factory=list)   # 融合后召回
    reranked: list[RetrievedItem] = field(default_factory=list)   # 重排后
    results: list[DiagnosisResult] = field(default_factory=list)  # 最终 Top3
    latency_ms: float = 0.0


class DiagnosisPipeline(Protocol):
    async def diagnose_with_trace(self, req: DiagnosisRequest) -> PipelineTrace: ...


@dataclass
class EvalReport:
    n: int
    recall_at_50: float
    mrr_at_10: float
    ndcg_at_10: float
    top3_hit: float
    top3_ci: tuple[float, float]
    top1_hit: float
    ece: float
    p99_latency_ms: float
    bad_cases: list[str]  # 未命中样本 ID，交给人工分析


class EvalRunner:
    def __init__(self, pipeline: DiagnosisPipeline, samples: list[GoldenSample]):
        self.pipeline = pipeline
        self.samples = samples

    async def run(self, concurrency: int = 8) -> EvalReport:
        sem = asyncio.Semaphore(concurrency)

        async def one(s: GoldenSample):
            async with sem:
                return s, await self.pipeline.diagnose_with_trace(s.request)

        pairs = await asyncio.gather(*(one(s) for s in self.samples))
        return self._aggregate(pairs)

    def _aggregate(self, pairs) -> EvalReport:
        n = len(pairs)
        r50 = mrr = ndcg = top3 = top1 = 0.0
        calib: list[tuple[float, bool]] = []
        latencies: list[float] = []
        bad: list[str] = []

        for s, t in pairs:
            gold = set(s.gold_root_causes)
            r50 += metrics.recall_at_k(t.recalled, gold, 50)
            mrr += metrics.mrr_at_k(t.reranked or t.recalled, gold, 10)
            ndcg += metrics.ndcg_at_k(t.reranked or t.recalled, gold, 10)
            hit3 = metrics.top_k_hit(t.results, gold, 3)
            top3 += hit3
            top1 += metrics.top_k_hit(t.results, gold, 1)
            calib.extend((r.confidence, r.root_cause_id in gold) for r in t.results)
            latencies.append(t.latency_ms)
            if not hit3:
                bad.append(s.sample_id)

        latencies.sort()
        p99 = latencies[int(0.99 * (n - 1))] if latencies else 0.0
        return EvalReport(
            n=n,
            recall_at_50=r50 / n,
            mrr_at_10=mrr / n,
            ndcg_at_10=ndcg / n,
            top3_hit=top3 / n,
            top3_ci=metrics.wilson_interval(int(top3), n),
            top1_hit=top1 / n,
            ece=metrics.expected_calibration_error(calib),
            p99_latency_ms=p99,
            bad_cases=bad,
        )
