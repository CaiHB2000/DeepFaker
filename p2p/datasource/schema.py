from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
import time

@dataclass
class RdapRecord:
    domain: str
    registry: Optional[str] = None
    domain_status: List[str] = field(default_factory=list)
    events: Dict[str, str] = field(default_factory=dict)  # e.g., {"registration": "...", "expiration": "..."}
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HttpFingerprint:
    url_requested: str
    final_url: str
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    server: Optional[str] = None
    via: Optional[str] = None
    content_type: Optional[str] = None

@dataclass
class TlsFingerprint:
    host: str
    port: int = 443
    sni: Optional[str] = None
    issuer: Optional[str] = None
    subject: Optional[str] = None
    not_before: Optional[str] = None
    not_after: Optional[str] = None
    serial_number: Optional[str] = None
    sig_hash_alg: Optional[str] = None
    subject_alternative_names: List[str] = field(default_factory=list)

@dataclass
class WebArchiveRecord:
    url: str
    first_seen_ts: Optional[str] = None   # CDX timestamp like "20200101123456"
    latest_seen_ts: Optional[str] = None  # optional
    snapshots: int = 0

@dataclass
class ProvenanceExternalMeta:
    url: str
    domain: str
    ts_collected: float
    rdap: Optional[RdapRecord] = None
    wayback: Optional[WebArchiveRecord] = None
    http: Optional[HttpFingerprint] = None
    tls: Optional[TlsFingerprint] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses nested serialization already handled by asdict
        return d

def now_unix() -> float:
    return time.time()

# ===== C2PA & Fingerprints =====
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class C2PAReport:
    path: str
    present: bool
    valid: Optional[bool] = None      # 若可解析验证结果则给出
    vendor: Optional[str] = None      # 发行方/实现者（若可解析）
    claim_generator: Optional[str] = None
    assertions_count: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class PDQResult:
    path: str
    available: bool
    hash_hex: Optional[str] = None     # 若能解析则填；否则为 None
    quality: Optional[float] = None    # 有些实现会给质量分
    raw_text: Optional[str] = None     # 解析不了时保留原始输出
    error: Optional[str] = None

@dataclass
class VPDQFrame:
    ts: float
    hash_hex: str
    quality: float

@dataclass
class VPDQResult:
    path: str
    available: bool
    frames: List[VPDQFrame] = field(default_factory=list)
    error: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None  # 某些CLI可直接吐JSON

@dataclass
class FileFingerprints:
    path: str
    pdq: Optional[PDQResult] = None
    vpdq: Optional[VPDQResult] = None
    tmk_path: Optional[str] = None   # TMK+PDQF 特征文件路径（若生成）
    errors: List[str] = field(default_factory=list)
