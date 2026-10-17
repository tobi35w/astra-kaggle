"""Typed results shared by acquisition, validation and reporting stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class DownloadUnitResult:
    symbol: str
    day: date
    status: str
    path: str = ""
    row_count: int = 0
    checksum: str = ""
    error: str = ""


@dataclass(frozen=True)
class DownloadSummary:
    requested: int
    downloaded: int
    cached: int
    missing: int
    failed: int
    invalid: int
    results: tuple[DownloadUnitResult, ...] = ()
    closed: int = 0
    deferred: int = 0

    @property
    def successful(self) -> bool:
        return (
            self.requested > 0
            and self.failed == 0
            and self.invalid == 0
            and self.missing == 0
            and self.deferred == 0
        )


@dataclass(frozen=True)
class Gap:
    previous: datetime
    next: datetime
    missing_minutes: int
    kind: str


@dataclass(frozen=True)
class ValidationReport:
    symbol: str
    timeframe: str
    rows: int
    duplicate_count: int
    conflicting_duplicate_count: int
    invalid_count: int
    out_of_order_count: int
    expected_gap_count: int
    suspicious_gaps: tuple[Gap, ...]
    earliest: datetime | None
    latest: datetime | None
    checksum: str = ""
    status: str = "FAIL"
    warnings: tuple[str, ...] = ()


@dataclass
class SymbolPipelineResult:
    symbol: str
    status: str
    downloads: DownloadSummary | None = None
    timeframes: dict[str, ValidationReport] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineSummary:
    dataset_version: str
    source: str
    start_date: date
    end_date: date
    symbols: tuple[SymbolPipelineResult, ...]
    manifest_path: str
    report_json_path: str
    report_text_path: str
    run_id: str = ""

    @property
    def successful(self) -> bool:
        return all(result.status == "PASS" for result in self.symbols)
