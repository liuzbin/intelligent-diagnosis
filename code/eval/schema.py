"""
评测集（黄金集）数据结构定义。

阶段 0 产出。所有后续阶段（检索调参、重排、换模型、微调）都在这套数据结构上做对比，
保证"同一把尺子"。

设计要点：
1. 以"标准根因类 root_cause_id"作为命中判定依据，而不是根因文本 —— 不同工程师写法不同。
2. 每条样本记录 closed_at（闭环时间），用于按时间切分，防止知识库与评测集泄漏。
3. 一条故障可能有多个可接受根因（例如"连接器虚焊"与"连接器氧化"被专家认为都对），
   所以 gold_root_causes 是列表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ProductLine(str, Enum):
    """产品线，用于分层抽样，保证评测集不被某一产品线主导。"""
    BASE_STATION = "base_station"   # 通讯基站设备
    PHONE = "phone"
    TABLET = "tablet"
    BOARD = "board"                 # 单板
    CHIP = "chip"


@dataclass
class DiagnosisRequest:
    """诊断请求：与线上接口的入参保持一致，评测时直接喂给线上链路。"""
    product_model: str
    product_line: ProductLine
    station: str                    # 工站
    test_item: str                  # 测试项
    error_code: Optional[str] = None
    measured_value: Optional[float] = None
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    device_id: Optional[str] = None
    symptom_text: str = ""          # 日志片段 / 现象描述


@dataclass
class GoldenSample:
    """黄金集样本。"""
    sample_id: str
    request: DiagnosisRequest
    gold_root_causes: list[str]                 # 可接受的标准根因类 ID（多个均算对）
    closed_at: datetime                         # 原工单闭环时间（时间切分依据）
    source_ticket_id: str                       # 原工单号（用于近重复剔除）
    annotators: list[str] = field(default_factory=list)  # 交叉标注人
    note: str = ""


@dataclass
class RetrievedItem:
    """检索 / 重排阶段的候选。"""
    chunk_id: str
    ticket_id: str
    root_cause_id: Optional[str]   # 候选所属工单的标准根因类
    score: float
    source: str = ""               # 来自哪一路召回：bm25 / dense / fused / rerank


@dataclass
class DiagnosisResult:
    """端到端输出：Top3 根因 + 置信度。"""
    root_cause_id: str
    confidence: float              # 0~1
    evidence_ticket_ids: list[str] = field(default_factory=list)
