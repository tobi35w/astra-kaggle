"""Frozen conditional USDJPY paper replays; cannot certify independent profitability."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.adapters import SQLiteRepository
from astra.historical.validation import _market_open
from astra.ml import Model
from astra.repositories import Entity, Journal, json_value
from scripts.hf_profitability_replay import simulate
from scripts.hf_profitability_statistics import paired_difference, summarize
from scripts.hf_source_clock import availability_view
from scripts.train_hf_forex_experiment import (
    DATA_ROOT,
    _load_ticker_rows,
    source_inventory,
    to_market_data,
)

REFERENCE = DATA_ROOT / "training" / "continuity" / "20261016T113345-01771fbe"
PROVENANCE = REPO / "data" / "astra-hf-profitability-provenance-20261016" / "audit.json"
MINUTE = timedelta(minutes=1)
COSTS = {"zero": (0.0, 0.0), "base": (1.0, 0.2), "double": (2.0, 0.4)}


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _hashes(paths):
    return {
        str(Path(p).resolve()): sha256(Path(p).read_bytes()).hexdigest() for p in paths
    }


def _implementation():
    return _hashes(
        REPO / p
        for p in (
            "scripts/evaluate_hf_profitability.py",
            "scripts/hf_profitability_replay.py",
            "scripts/hf_profitability_statistics.py",
            "scripts/hf_source_clock.py",
            "scripts/train_hf_forex_experiment.py",
            "astra/ml.py",
            "astra/models.py",
            "astra/data.py",
            "astra/indicators.py",
            "astra/risk.py",
            "astra/config.py",
            "astra/execution.py",
            "astra/strategies.py",
            "astra/repositories.py",
            "astra/adapters.py",
            "astra/historical/validation.py",
        )
    )


def _model(payload):
    value = dict(payload)
    for k in ("trained_at", "training_end", "validation_start"):
        value[k] = datetime.fromisoformat(value[k])
    for k in ("means", "scales", "coefficients", "feature_schema"):
        value[k] = tuple(value[k])
    return Model(**value)


def validate_provenance(audit, source):
    """Require the executed, revision-bound sample audit before shifting clocks."""
    if (
        audit.get("schema") != "astra.hf.sampled_provenance.v1"
        or audit.get("status") != "completed_sampled_comparison"
        or audit.get("dataset_id") != source.get("dataset_id")
        or audit.get("dataset_revision") != source.get("cache_snapshot")
        or audit.get("price_basis") != "bid"
        or audit.get("timestamp_convention") != "candle_open"
        or audit.get("availability_shift_seconds") != 60
        or audit.get("whole_cache_authenticated") is not False
    ):
        raise ValueError(
            "a matching sampled BID/open timestamp provenance audit is required"
        )
    samples = [s for s in audit.get("samples", []) if s.get("pair") == "USDJPY"]
    if not samples or any(
        not isinstance(s.get("count"), int)
        or s["count"] < 1
        or s.get("zero_offset_ohlc_exact_matches") != s["count"]
        or s.get("minus_one_minute_ohlc_matches") != 0
        or s.get("plus_one_minute_ohlc_matches") != 0
        for s in samples
    ):
        raise ValueError("USDJPY timestamp sample evidence is missing or inconsistent")


def _availability_view(raw):
    """Derive availability from corroborated source-open timestamps; retain originals."""
    return availability_view(raw)


def _windows(data, start):
    end = data.candles[-1].timestamp
    stamps = {c.timestamp for c in data.candles}
    months = sorted(
        {
            (c.timestamp.year, c.timestamp.month)
            for c in data.candles
            if c.timestamp >= start
        }
    )
    windows = [{"name": "continuous", "scope": "primary", "start": start, "end": end}]
    omitted = []
    for year, month in months:
        first = datetime(year, month, 1, tzinfo=UTC)
        following = datetime(
            year + int(month == 12), 1 if month == 12 else month + 1, 1, tzinfo=UTC
        )
        last = min(end, following - MINUTE)
        # Calendar is known before prices; never search observed data for a convenient exit.
        while not _market_open(last - MINUTE):
            last -= MINUTE
        first = max(first, start)
        name = f"{year}-{month:02d}"
        if last not in stamps or last <= first:
            omitted.append(
                {
                    "month": name,
                    "required_end": last.isoformat(),
                    "reason": "predeclared calendar/session endpoint unavailable",
                }
            )
        else:
            windows.append(
                {
                    "name": name,
                    "scope": "independent_month",
                    "start": first,
                    "end": last,
                }
            )
    return windows, omitted


def _case_summary(identity, window, basis, cost_name, policy, result, metrics):
    return {
        "case_id": identity,
        "scope": window["scope"],
        "window": window["name"],
        "basis": basis,
        "cost": cost_name,
        "policy": policy,
        "metrics": metrics,
        "gaps": {k: v for k, v in result["gaps"].items() if k != "events"},
        "risk_halted": result["limits"]["permanent_halt"],
        "rejection_reasons": result["rejection_reasons"],
        "file": identity + ".json",
    }


def _assessment(cases, omitted):
    assessments = []
    for case in cases:
        if (
            case["scope"] != "primary"
            or case["policy"] != "model"
            or case["cost"] == "zero"
        ):
            continue
        m = case["metrics"]
        failures = []
        if m["net_pnl"] <= 0:
            failures.append("non-positive net return after assumed costs")
        if m["trade_count"] < 100 or m["active_days"] < 20:
            failures.append("fewer than 100 trades or 20 active observed days")
        if m["replay_max_drawdown_pct"] > 10:
            failures.append("marked drawdown exceeds 10%")
        if m["bootstrap_mean_daily_return"]["low_95"] <= 0:
            failures.append("descriptive daily-return lower bound is not positive")
        if case["paired_control"]["net_pnl_difference"] <= 0:
            failures.append("does not outperform the training-majority control")
        if any(case["gaps"].values()):
            failures.append("unobserved entry or exit prices across a data gap")
        monthly = [
            c
            for c in cases
            if c["scope"] == "independent_month"
            and c["policy"] == "model"
            and c["basis"] == case["basis"]
            and c["cost"] == case["cost"]
        ]
        if (
            len(monthly) < 3
            or sum(c["metrics"]["net_pnl"] > 0 for c in monthly) / max(1, len(monthly))
            < 0.7
            or omitted
        ):
            failures.append(
                "insufficient complete monthly tests or fewer than 70% positive months"
            )
        assessments.append(
            {
                "case_id": case["case_id"],
                "passed_conditional_criteria": not failures,
                "unmet_criteria": failures,
            }
        )
    return {
        "status": "CONDITIONAL_CRITERIA_MET"
        if assessments and all(c["passed_conditional_criteria"] for c in assessments)
        else "PROFITABILITY_NOT_ESTABLISHED",
        "cases": assessments,
        "profitability_proven": False,
        "promotion_eligible": False,
        "external_requirements": [
            "independently reserved unused data",
            "full price and timestamp provenance",
            "credible observed trading costs and execution validation",
        ],
    }


def _markdown(summary):
    lines = [
        "# USDJPY conditional profitability experiment",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "All cash amounts are JPY. Each independent monthly test starts with its own 1,000,000 JPY account; their returns are not a compounded portfolio. The continuous case retains all risk history. BID is supported by sampled provenance; MID is a price-basis sensitivity case.",
        "",
        "| Window | Basis | Costs | Policy | Trades | Net JPY | Return | Marked drawdown | Risk halted |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for c in summary["cases"]:
        m = c["metrics"]
        lines.append(
            f"| {c['window']} | {c['basis']} | {c['cost']} | {c['policy']} | {m['trade_count']} | {m['net_pnl']:.2f} | {m['total_return_pct']:.4f}% | {m['replay_max_drawdown_pct']:.4f}% | {c['risk_halted']} |"
        )
    lines += [
        "",
        "Base costs are a one-pip spread and 0.2-pip slippage each side; doubled costs double both. Commission is zero. These are assumed scenarios, not observed broker costs. No margin, financing or live execution claim is made. A risk halt is never reset in the continuous case.",
        "",
        "Missing minutes can hide fills and stops. Pending orders crossing gaps are cancelled with uncertainty recorded; open positions exit at the first observed open with costs, not a fabricated earlier price. Such cases cannot establish profitability. A five-loss halt may leave too few trades to assess the model.",
        "",
        "The original selected model weights are frozen. Source timestamps are shifted from open to availability by one minute using the separate provenance audit. Decisions use trailing observed candles only; future-label continuity is not an entry filter. Model creation time is preserved, and results are retrospective development evidence.",
        "",
        "Per-case JSON files contain trades, order events, both risk approvals, daily equity, reconciliation, and descriptive five-observed-day block-bootstrap intervals. Source/code/reference identities and Journal payloads must pass verification before completion.",
        "",
        f"Assessment: {summary.get('assessment', {}).get('status', 'unfinished')}. Profitability is not independently proven.",
        "",
    ]
    if summary.get("error"):
        lines += ["Failure: " + summary["error"], ""]
    return "\n".join(lines)


def run_experiment(
    output_root,
    *,
    reference_root=REFERENCE,
    provenance_path=PROVENANCE,
    bootstrap_replicates=2000,
):
    reference_root = Path(reference_root)
    ref_files = [
        reference_root / name
        for name in (
            "plan.json",
            "comparison.json",
            "USDJPY-M1-report.json",
            "verification.json",
        )
    ]
    ref_files.append(Path(provenance_path))
    reference_hashes = _hashes(ref_files)
    source = source_inventory()
    validate_provenance(
        json.loads(Path(provenance_path).read_text(encoding="utf-8")), source
    )
    reference_plan = json.loads(ref_files[0].read_text(encoding="utf-8"))
    prior = json.loads(ref_files[1].read_text(encoding="utf-8"))
    report = json.loads(ref_files[2].read_text(encoding="utf-8"))
    verified = json.loads(ref_files[3].read_text(encoding="utf-8"))
    if (
        reference_plan["source"] != source
        or prior["status"] != "COMPLETED"
        or verified["status"] != "VERIFIED"
        or report["status"] != "TRAINED"
    ):
        raise ValueError("source and verified frozen reference must match")
    if (
        report["model"]
        != next(r for r in prior["reports"] if r["pair"] == "USDJPY")["model"]
    ):
        raise ValueError("frozen model differs between reference reports")
    reference_shift = report.get("source_timestamp_shift_minutes", 0)
    if (
        reference_shift not in (0, 1)
        or reference_plan.get("source_timestamp_shift_minutes", 0) != reference_shift
    ):
        raise ValueError(
            "reference timestamp convention is unsupported or inconsistent"
        )
    rows = _load_ticker_rows("USDJPY", reference_plan["maximum_rows_per_pair"])
    raw = to_market_data(
        rows, "USDJPY", max_rows=reference_plan["maximum_rows_per_pair"]
    )
    del rows
    raw_version = raw.metadata["dataset_version"]
    if raw_version != report.get("raw_dataset_version", report["dataset_version"]):
        raise ValueError("selected source rows differ from the frozen model experiment")
    data = _availability_view(raw)
    if (
        reference_shift
        and data.metadata["dataset_version"] != report["dataset_version"]
    ):
        raise ValueError(
            "availability source rows differ from the frozen model experiment"
        )
    del raw
    model = _model(report["model"])
    remaining_shift = (1 - reference_shift) * MINUTE
    start = model.validation_start + remaining_shift
    if model.training_end + remaining_shift >= start:
        raise ValueError("corrected training label overlaps validation")
    windows, omitted = _windows(data, start)
    code = _implementation()
    recorded = datetime.now(UTC)
    identity = f"{recorded:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    output = Path(output_root).resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "run_id": identity,
        "recorded_at": recorded.isoformat(),
        "pair": "USDJPY",
        "source": source,
        "raw_dataset_version": raw_version,
        "availability_dataset_version": data.metadata["dataset_version"],
        "reference_sha256": reference_hashes,
        "implementation_sha256": code,
        "source_timestamp_shift_minutes": 1,
        "reference_timestamp_shift_minutes": reference_shift,
        "timestamp_evidence": str(provenance_path),
        "model": report["model"],
        "selection_bias": "USDJPY selected after inspecting prior four-pair classification results; reused development history",
        "effective_training_label_end": (
            model.training_end + remaining_shift
        ).isoformat(),
        "windows": json_value(windows),
        "omitted_months": omitted,
        "policies": ["model", "training_majority"],
        "threshold": 0.5,
        "price_basis_scenarios": ["bid", "mid"],
        "costs_spread_slippage_pips": COSTS,
        "stop_atr": 2,
        "target_atr": 4,
        "time_exit_minutes": 3,
        "initial_equity_each_account": 1000000,
        "account_currency": "JPY",
        "maximum_base_units": 10000,
        "risk_fraction": 0.005,
        "daily_loss_limit": 0.05,
        "drawdown_limit": 0.20,
        "consecutive_loss_halt": 5,
        "training_up_fraction": report["validation"]["overall"][
            "constant_probability_baseline"
        ],
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": 20261016,
            "observed_day_block_size": 5,
        },
        "conditional_criteria": [
            "positive net at base and double costs under both bases",
            "outperform majority control",
            ">=100 trades and >=20 active days per continuous case",
            "marked drawdown<=10%",
            "positive descriptive mean daily return lower bound",
            "at least 3 independent months and >=70% positive; none omitted",
            "no uncertain fills/exits across gaps",
        ],
        "promotion_eligible": False,
        "profitability_proven": False,
    }
    _write(output / "plan.json", plan)
    repo = SQLiteRepository(str(output / "experiment.sqlite3"))
    journal = Journal(repo)
    summary = {
        "run_id": identity,
        "status": "RUNNING",
        "output_directory": str(output),
        "cases": [],
        "omitted_months": omitted,
        "source_identity_rechecked": False,
        "implementation_identity_rechecked": False,
        "reference_identity_rechecked": False,
        "journal_payloads_verified": False,
        "profitability_proven": False,
        "promotion_eligible": False,
    }
    expected_records = {}
    try:
        journal.record(
            Entity.AUDIT, recorded, {"event": "hf_profitability_planned", **plan}
        )
        journal.record(Entity.MODEL, recorded, model, pair="USDJPY", timeframe="M1")
        for window in windows:
            for basis in ("bid", "mid"):
                for cost_name, (spread, slippage) in COSTS.items():
                    paired = {}
                    for policy in ("model", "training_majority"):
                        case_id = f"{window['name']}-{basis}-{cost_name}-{policy}"
                        print(f"Replaying {case_id} ...", flush=True)
                        result = simulate(
                            data,
                            model,
                            start_at=window["start"],
                            end_at=window["end"],
                            price_basis=basis,
                            spread_pips=spread,
                            slippage_pips=slippage,
                            policy=policy,
                            training_up_fraction=plan["training_up_fraction"],
                        )
                        metrics = summarize(
                            result, bootstrap_replicates=bootstrap_replicates
                        )
                        if not result["reconciliation"]["passed"]:
                            raise ValueError("paper ledger reconciliation failed")
                        payload = {
                            "case_id": case_id,
                            "scope": window["scope"],
                            "result": result,
                            "metrics": metrics,
                        }
                        _write(output / (case_id + ".json"), payload)
                        record = journal.record(
                            Entity.EVALUATION,
                            datetime.now(UTC),
                            payload,
                            pair="USDJPY",
                            timeframe="M1",
                            strategy=policy,
                        )
                        expected_records[record.record_id] = sha256(
                            json.dumps(
                                payload, sort_keys=True, allow_nan=False
                            ).encode()
                        ).hexdigest()
                        case = _case_summary(
                            case_id, window, basis, cost_name, policy, result, metrics
                        )
                        summary["cases"].append(case)
                        paired[policy] = result
                        print(
                            f"{case_id}: {metrics['trade_count']} trades, {metrics['net_pnl']:.2f} JPY",
                            flush=True,
                        )
                    model_case = summary["cases"][-2]
                    model_case["paired_control"] = paired_difference(
                        paired["model"],
                        paired["training_majority"],
                        bootstrap_replicates=bootstrap_replicates,
                    )
        if source_inventory() != source:
            raise ValueError("source changed during replay")
        summary["source_identity_rechecked"] = True
        if _implementation() != code:
            raise ValueError("implementation changed during replay")
        summary["implementation_identity_rechecked"] = True
        if _hashes(ref_files) != reference_hashes:
            raise ValueError("reference evidence changed during replay")
        summary["reference_identity_rechecked"] = True
        saved = repo.query(Entity.EVALUATION)
        if len(saved) != len(expected_records) or any(
            sha256(
                json.dumps(r.payload, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            != expected_records.get(r.record_id)
            for r in saved
        ):
            raise ValueError("Journal evaluation payloads differ from saved scenarios")
        summary["journal_payloads_verified"] = True
        summary["assessment"] = _assessment(summary["cases"], omitted)
        summary["status"] = "COMPLETED"
    except KeyboardInterrupt:
        summary.update(
            status="INTERRUPTED", error="replay interrupted; completed cases retained"
        )
    except Exception as error:  # noqa: BLE001 - retain honest failed-run evidence.
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    finally:
        summary["completed_at"] = datetime.now(UTC).isoformat()
        try:
            record = journal.record(
                Entity.AUDIT,
                datetime.fromisoformat(summary["completed_at"]),
                {"event": "hf_profitability_finished", **summary},
            )
            if not any(
                r.record_id == record.record_id and r.payload == record.payload
                for r in repo.query(Entity.AUDIT)
            ):
                raise ValueError("final audit did not roundtrip")
        except Exception as error:  # noqa: BLE001 - preserve failure receipt if Journal fails.
            summary.update(status="FAILED", error=f"final audit failed: {error}")
        finally:
            repo.close()
        _write(output / "summary.json", summary)
        (output / "report.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    parser.add_argument(
        "--output-root", type=Path, default=DATA_ROOT / "training" / "profitability"
    )
    args = parser.parse_args(argv)
    result = run_experiment(
        args.output_root,
        reference_root=args.reference_root,
        provenance_path=args.provenance,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(Path(result["output_directory"]) / "report.md"),
            }
        )
    )
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
