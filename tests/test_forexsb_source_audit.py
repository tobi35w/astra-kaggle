"""Offline ForexSB audit tests; every market observation here is synthetic."""

import gzip
import hashlib
import json
import lzma
import struct
from datetime import UTC, date, datetime, timedelta

import pytest

from astra.adapters import SQLiteRepository
from astra.historical.downloader import _file
from astra.repositories import Entity
from scripts import audit_forexsb_source as audit

DAY = date(2024, 1, 8)  # A complete Monday inside the existing FX calendar.
START = datetime(2024, 1, 8, tzinfo=UTC)
AT = datetime(2024, 1, 10, tzinfo=UTC)
OFFSET = int((START - audit.EPOCH) / audit.MINUTE)


def _row(minute=0, *, scale=100000, volume=7):
    return (OFFSET + minute, scale, scale + 2, scale - 1, scale + 1, volume, 3)


def _archive(rows):
    return gzip.compress(b"".join(audit.RECORD.pack(*row) for row in rows), mtime=0)


def _metadata(symbol="EURUSD", scale=100000):
    return {"symbol": symbol, "priceScale": scale, "volumeScale": 10}


def test_decoder_uses_availability_time_and_preserves_raw_volume():
    candles = audit.decode_archive(
        _archive([_row(volume=0), _row(1, volume=17)]), "EURUSD", _metadata()
    )
    assert candles[0].timestamp == START + timedelta(minutes=1)
    assert candles[1].timestamp == START + timedelta(minutes=2)
    assert candles[0].open == 1
    assert candles[0].high == 1.00002
    assert candles[0].low == 0.99999
    assert candles[0].close == 1.00001
    # The website substitutes 1 for zero and rounds after volumeScale division.
    # Research retains the source field and never invents that volume.
    assert [c.volume for c in candles] == [0, 17]
    assert {c.volume_type for c in candles} == {"provider"}
    jpy = audit.decode_archive(
        _archive([_row(scale=1000)]), "USDJPY", _metadata("USDJPY", 1000)
    )
    assert jpy[0].high == 1.002


@pytest.mark.parametrize(
    "rows",
    [
        [_row(), _row()],  # Identical duplicate must not be silently deduplicated.
        [_row(), _row(volume=99)],  # Conflicting duplicate.
        [_row(1), _row()],  # Out of order; never sort before validating.
    ],
)
def test_decoder_rejects_duplicate_or_reversed_timestamps(rows):
    with pytest.raises(ValueError, match="ordered and unique"):
        audit.decode_archive(_archive(rows), "EURUSD", _metadata())


@pytest.mark.parametrize("field,value", [(2, 1), (3, 0), (5, -1), (6, -1)])
def test_decoder_rejects_invalid_prices_volume_and_spread(field, value):
    row = list(_row())
    row[field] = value
    with pytest.raises(ValueError, match="invalid OHLC|negative volume or spread"):
        audit.decode_archive(_archive([row]), "EURUSD", _metadata())


@pytest.mark.parametrize(
    "metadata", [_metadata("GBPUSD"), _metadata(scale=1000), _metadata(scale=True)]
)
def test_decoder_rejects_mismatched_symbol_or_scale(metadata):
    with pytest.raises(ValueError, match="inconsistent source price scale"):
        audit.decode_archive(_archive([_row()]), "EURUSD", metadata)


@pytest.mark.parametrize(
    "payload",
    [
        b"not gzip",
        _archive([_row()])[:-4],
        gzip.compress(struct.pack("<6i", *_row()[:6])),  # Legacy format unsupported.
        gzip.compress(b""),
    ],
)
def test_decoder_rejects_corrupt_truncated_and_unsupported_archives(payload):
    with pytest.raises(ValueError, match="gzip integrity|record layout"):
        audit.decode_archive(payload, "EURUSD", _metadata())


def test_decoder_bounds_decompression(monkeypatch):
    monkeypatch.setattr(audit, "MAX_DECOMPRESSED", audit.RECORD.size)
    with pytest.raises(ValueError, match="unsupported size"):
        audit.decode_archive(_archive([_row(), _row(1)]), "EURUSD", _metadata())


def _write_sources(tmp_path, *, missing_minute=None):
    source = tmp_path / "synthetic-source"
    reference = tmp_path / "synthetic-reference"
    source.mkdir()
    (reference / "manifests").mkdir(parents=True)
    info, entries = {}, {}
    for symbol in audit.PAIRS:
        scale = audit.price_scale_for(symbol)
        info[symbol] = _metadata(symbol, scale)
        # Both edge sentinels must be excluded from the requested source day.
        rows = [_row(i, scale=scale) for i in range(-1, 1441) if i != missing_minute]
        (source / f"forexsb-{symbol}1.lb.gz").write_bytes(_archive(rows))
        raw = lzma.compress(
            b"".join(
                struct.pack(
                    ">IIIIIf", i * 60, scale, scale + 1, scale - 1, scale + 2, 7.0
                )
                for i in range(1440)
            )
        )
        path = _file(reference, symbol, DAY)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        entries[f"{symbol}:{DAY}"] = {
            "status": "downloaded",
            "path": str(path.relative_to(reference)),
            "checksum": hashlib.sha256(raw).hexdigest(),
            "rows": 1440,
        }
    (source / "forexsb-info.json").write_text(json.dumps(info), encoding="utf-8")
    (reference / "manifests" / "dukascopy_manifest.json").write_text(
        json.dumps({"downloads": entries}), encoding="utf-8"
    )
    (reference / "acquisition-summary.json").write_text(
        '{"evidence_kind":"SYNTHETIC_TEST_ONLY"}', encoding="utf-8"
    )
    return source, reference


def test_audit_preserves_inputs_clips_causally_and_records_its_journal(tmp_path):
    source, reference = _write_sources(tmp_path, missing_minute=3)
    originals = {
        p: p.read_bytes()
        for root in (source, reference)
        for p in root.rglob("*")
        if p.is_file()
    }
    output, report = audit.run_audit(
        source, reference, tmp_path / "audit-results", DAY, DAY, at=AT
    )
    assert report["replacement_approved"] is False
    assert report["managed_dataset_import_performed"] is False
    assert report["training_or_evaluation_performed"] is False
    assert report["inputs_unchanged_after_audit"] is True
    assert {p: p.read_bytes() for p in originals} == originals
    for pair in report["pairs"]:
        assert pair["source_rows"] == 1441
        assert pair["scope_rows"] == 1439
        assert pair["missing_open_minutes"] == 1
        assert pair["missing_open_utc"] == ["2024-01-08T00:03:00+00:00"]
        assert pair["observed_open_days"] == 1
        assert pair["fully_complete_open_days"] == 0
        assert pair["timeframes"]["M1"]["status"] == "WARN"
        assert pair["timeframes"]["M1"]["latest"] == "2024-01-09T00:00:00+00:00"
        assert pair["timeframes"]["M5"]["rows"] == 287
        assert pair["timeframes"]["M5"]["earliest"] == "2024-01-08T00:10:00+00:00"
        assert pair["reference_overlap"]["matched_minutes"] == 1439
        assert pair["reference_overlap"]["exact_ohlc_rows"] == 1439
        assert pair["reference_overlap"]["omitted_reference_other_minutes"] == 1
    repository = SQLiteRepository(str(output / "audit.sqlite3"))
    try:
        records = repository.query(Entity.AUDIT)
        assert len(records) == 1
        assert records[0].record_id == report["journal_record_id"]
        assert records[0].occurred_at == AT
        assert records[0].payload["replacement_approved"] is False
    finally:
        repository.close()
    assert not (output / "manifests").exists()


def test_complete_fixture_is_not_automatically_admitted_or_overwritten(tmp_path):
    source, reference = _write_sources(tmp_path)
    output, report = audit.run_audit(
        source, reference, tmp_path / "audit-results", DAY, DAY, at=AT
    )
    before = (output / "audit.json").read_bytes()
    for pair in report["pairs"]:
        assert pair["missing_open_minutes"] == 0
        assert pair["fully_complete_open_days"] == 1
        assert {frame["status"] for frame in pair["timeframes"].values()} == {"PASS"}
    assert report["replacement_approved"] is False
    repeated, _ = audit.run_audit(
        source, reference, tmp_path / "audit-results", DAY, DAY, at=AT
    )
    assert repeated != output
    assert (output / "audit.json").read_bytes() == before


def test_reference_tampering_fails_before_any_output_is_written(tmp_path):
    source, reference = _write_sources(tmp_path)
    _file(reference, "EURUSD", DAY).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.run_audit(source, reference, tmp_path / "outputs", DAY, DAY, at=AT)
    assert not (tmp_path / "outputs").exists()


def test_output_cannot_target_reference_and_future_windows_are_rejected(tmp_path):
    source, reference = _write_sources(tmp_path)
    with pytest.raises(ValueError, match="separate from the reference"):
        audit.run_audit(source, reference, reference / "audit", DAY, DAY, at=AT)
    with pytest.raises(ValueError, match="completed source date range"):
        audit.run_audit(source, reference, tmp_path / "outputs", DAY, DAY, at=START)
    assert not (tmp_path / "outputs").exists()
