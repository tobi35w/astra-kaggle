"""Summarize loss drivers from the completed curated historical campaign."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from uuid import NAMESPACE_URL, uuid5

from astra.adapters import SQLiteRepository
from astra.repositories import Entity, Journal

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "training-arena" / "curated-history-h1-h4-v1"
RESULTS_PATH = OUTPUT_ROOT / "results.jsonl"
TRADES_PATH = OUTPUT_ROOT / "trades.jsonl"
MODELS_PATH = OUTPUT_ROOT / "models.jsonl"
DIAGNOSTICS_PATH = OUTPUT_ROOT / "loss-diagnostics.json"
MARKDOWN_PATH = OUTPUT_ROOT / "loss-diagnostics.md"
JOURNAL_PATH = ROOT / "data" / "training-arena" / "arena-journal.sqlite3"
CAMPAIGN_ID = "astra-curated-four-pair-windows-h1-h4-v1"
DIAGNOSTIC_ID = str(uuid5(NAMESPACE_URL, f"{CAMPAIGN_ID}:loss-diagnostics:v2"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize_results(results: list[dict]) -> dict:
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    by_pair_frame: dict[str, list[dict]] = defaultdict(list)
    paired: dict[tuple[str, str, str, str], dict[str, dict]] = defaultdict(dict)

    for row in results:
        by_scenario[row["cost_scenario"]].append(row)
        by_strategy[f"{row['strategy']}|{row['cost_scenario']}"].append(row)
        by_pair_frame[
            f"{row['pair']}|{row['timeframe']}|{row['cost_scenario']}"
        ].append(row)
        pair_key = (row["pair"], row["timeframe"], row["window_id"], row["strategy"])
        paired[pair_key][row["cost_scenario"]] = row

    def run_stats(rows: list[dict]) -> dict:
        returns = [float(row["net_return_pct"]) for row in rows]
        return {
            "run_count": len(rows),
            "positive_runs": sum(value > 0 for value in returns),
            "negative_runs": sum(value < 0 for value in returns),
            "flat_runs": sum(value == 0 for value in returns),
            "mean_return_pct": mean(returns) if returns else 0.0,
            "median_return_pct": median(returns) if returns else 0.0,
            "trade_count": sum(int(row["trade_count"]) for row in rows),
            "runs_below_30_trades": sum(int(row["trade_count"]) < 30 for row in rows),
        }

    cost_deltas = []
    base_to_loss = 0
    double_to_gain = 0
    for pair_key, scenarios in paired.items():
        if "base" not in scenarios or "double_cost" not in scenarios:
            continue
        base = float(scenarios["base"]["net_return_pct"])
        doubled = float(scenarios["double_cost"]["net_return_pct"])
        cost_deltas.append(doubled - base)
        base_to_loss += base > 0 >= doubled
        double_to_gain += base <= 0 < doubled

    return {
        "by_cost_scenario": {
            key: run_stats(rows) for key, rows in sorted(by_scenario.items())
        },
        "by_strategy_and_cost": {
            key: run_stats(rows) for key, rows in sorted(by_strategy.items())
        },
        "by_pair_timeframe_and_cost": {
            key: run_stats(rows) for key, rows in sorted(by_pair_frame.items())
        },
        "paired_cost_sensitivity": {
            "paired_run_count": len(cost_deltas),
            "median_return_change_double_minus_base_pp": median(cost_deltas)
            if cost_deltas
            else 0.0,
            "mean_return_change_double_minus_base_pp": mean(cost_deltas)
            if cost_deltas
            else 0.0,
            "positive_base_to_nonpositive_double": int(base_to_loss),
            "nonpositive_base_to_positive_double": int(double_to_gain),
        },
    }


def summarize_trades(trades: list[dict]) -> dict:
    scenarios: dict[str, list[dict]] = defaultdict(list)
    strategy_scenarios: dict[str, list[dict]] = defaultdict(list)
    for row in trades:
        experiment = row["payload"]["experiment_id"]
        scenario = experiment.split(":", 1)[0]
        scenarios[scenario].append(row)
        strategy_scenarios[f"{row['strategy']}|{scenario}"].append(row)

    def trade_stats(rows: list[dict]) -> dict:
        reasons: dict[str, int] = defaultdict(int)
        regimes: dict[str, list[tuple[float, int]]] = defaultdict(list)
        sides: dict[str, list[tuple[float, int]]] = defaultdict(list)
        r_values = []
        wins = losses = flat = 0
        for row in rows:
            trade = row["payload"]["trade"]
            pnl = float(row["payload"].get("pnl", 0.0))
            risk = float(trade.get("risk_per_unit", 0.0)) * float(
                trade.get("size", 0.0)
            )
            outcome_r = pnl / risk if risk else 0.0
            r_values.append(outcome_r)
            wins += pnl > 0
            losses += pnl < 0
            flat += pnl == 0
            reason = str(trade.get("reason", "unknown"))
            reasons[reason] += 1
            regimes[str(trade.get("regime", "unknown"))].append((outcome_r, pnl))
            sides[str(trade.get("side", "unknown"))].append((outcome_r, pnl))

        def segment_stats(segments: dict[str, list[tuple[float, int]]]) -> dict:
            result = {}
            for name, values in sorted(segments.items()):
                segment_rs = [item[0] for item in values]
                segment_wins = sum(item[1] > 0 for item in values)
                result[name] = {
                    "trades": len(values),
                    "wins": segment_wins,
                    "losses": sum(item[1] < 0 for item in values),
                    "win_rate": segment_wins / len(values) if values else 0.0,
                    "mean_r": mean(segment_rs) if segment_rs else 0.0,
                }
            return result

        return {
            "trade_count": len(rows),
            "wins": wins,
            "losses": losses,
            "flat": flat,
            "win_rate": wins / len(rows) if rows else 0.0,
            "mean_r": mean(r_values) if r_values else 0.0,
            "median_r": median(r_values) if r_values else 0.0,
            "exit_reasons": dict(sorted(reasons.items())),
            "by_regime": segment_stats(regimes),
            "by_side": segment_stats(sides),
        }

    return {
        "by_cost_scenario": {
            scenario: trade_stats(rows) for scenario, rows in sorted(scenarios.items())
        },
        "by_strategy_and_cost": {
            identity: trade_stats(rows)
            for identity, rows in sorted(strategy_scenarios.items())
        },
    }


def summarize_ml_confidence(trades: list[dict]) -> dict:
    bins = ((0.55, 0.60), (0.60, 0.65), (0.65, 1.01))
    result = {}
    for scenario in ("base", "double_cost"):
        result[scenario] = {}
        for low, high in bins:
            values = []
            for row in trades:
                if row.get("strategy") != "ml_direction":
                    continue
                if not row["payload"]["experiment_id"].startswith(f"{scenario}:"):
                    continue
                trade = row["payload"]["trade"]
                confidence = float(trade.get("confidence", 0.0))
                if low <= confidence < high:
                    pnl = float(row["payload"].get("pnl", 0.0))
                    risk = float(trade.get("risk_per_unit", 0.0)) * float(
                        trade.get("size", 0.0)
                    )
                    values.append((pnl, pnl / risk if risk else 0.0))
            result[scenario][f"{low:.2f}_{high:.2f}"] = {
                "trades": len(values),
                "wins": sum(pnl > 0 for pnl, _ in values),
                "mean_r": mean(r for _, r in values) if values else 0.0,
                "median_r": median(r for _, r in values) if values else 0.0,
            }
    return result


def build_diagnostics(
    results: list[dict], trades: list[dict], models: list[dict]
) -> dict:
    trained = [row for row in models if row.get("status") == "TRAINED"]
    validations = [row["validation"] for row in trained if row.get("validation")]
    return {
        "campaign_id": CAMPAIGN_ID,
        "diagnostic_version": 2,
        "diagnostic_id": DIAGNOSTIC_ID,
        "created_at": datetime.now(UTC),
        "source": "local curated historical campaign outputs",
        "result_count": len(results),
        "trade_record_count": len(trades),
        "trained_model_count": len(trained),
        "run_diagnostics": summarize_results(results),
        "trade_diagnostics": summarize_trades(trades),
        "ml_confidence_bins": summarize_ml_confidence(trades),
        "model_validation": {
            "mean_accuracy": mean(
                float(row["validation_accuracy"]) for row in validations
            )
            if validations
            else None,
            "median_accuracy": median(
                float(row["validation_accuracy"]) for row in validations
            )
            if validations
            else None,
            "mean_brier": mean(float(row["validation_brier"]) for row in validations)
            if validations
            else None,
        },
        "promotion_status": "NOT_QUALIFIED",
        "promotion_reason": "Run returns average negative in both cost scenarios; confidence buckets do not show a sufficiently large, stable edge; many strategy-window samples are small.",
        "limitations": [
            "The cost cases use fixed assumed spread/slippage and zero commission.",
            "USD and JPY trade cash amounts are not summed together; trade outcomes are also shown in risk units.",
            "This diagnostic uses the same holdouts as the completed campaign and cannot serve as a fresh confirmatory evaluation.",
            "Do not promote any research model from this report into active inference.",
        ],
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Astra curated history loss diagnosis",
        "",
        f"Campaign: `{report['campaign_id']}`",
        f"Created: `{report['created_at'].isoformat()}`",
        f"Runs: {report['result_count']} · trained models: {report['trained_model_count']} · trade records: {report['trade_record_count']}",
        "",
        "## Run results by cost",
        "",
        "| Cost | Runs | Positive | Negative | Flat | Mean return | Trades | Under 30 trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in report["run_diagnostics"]["by_cost_scenario"].items():
        lines.append(
            f"| {scenario} | {values['run_count']} | {values['positive_runs']} | {values['negative_runs']} | {values['flat_runs']} | {values['mean_return_pct']:.3f}% | {values['trade_count']} | {values['runs_below_30_trades']} |"
        )
    lines.extend(
        [
            "",
            "## Trade outcomes",
            "",
            "| Cost | Trades | Wins | Losses | Win rate | Mean R | Stop loss | Take profit |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, values in report["trade_diagnostics"]["by_cost_scenario"].items():
        reasons = values["exit_reasons"]
        lines.append(
            f"| {scenario} | {values['trade_count']} | {values['wins']} | {values['losses']} | {values['win_rate']:.1%} | {values['mean_r']:.3f} | {reasons.get('stop_loss', 0)} | {reasons.get('take_profit', 0)} |"
        )
    lines.extend(
        [
            "",
            "## ML confidence check",
            "",
            "| Cost | Confidence band | Trades | Wins | Mean R | Median R |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scenario, bands in report["ml_confidence_bins"].items():
        for band, values in bands.items():
            lines.append(
                f"| {scenario} | {band.replace('_', '–')} | {values['trades']} | {values['wins']} | {values['mean_r']:.3f} | {values['median_r']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Strategy trade outcomes",
            "",
            "| Strategy | Cost | Trades | Win rate | Mean R | Stop exits |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key, values in report["trade_diagnostics"]["by_strategy_and_cost"].items():
        strategy, scenario = key.split("|", 1)
        stop_exits = values["exit_reasons"].get("stop_loss", 0)
        lines.append(
            f"| {strategy} | {scenario} | {values['trade_count']} | {values['win_rate']:.1%} | {values['mean_r']:.3f} | {stop_exits} |"
        )
    lines.extend(
        [
            "",
            "## Trade outcomes by regime",
            "",
            "| Cost | Regime | Trades | Win rate | Mean R |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for scenario, values in report["trade_diagnostics"]["by_cost_scenario"].items():
        for regime, stats in values["by_regime"].items():
            lines.append(
                f"| {scenario} | {regime} | {stats['trades']} | {stats['win_rate']:.1%} | {stats['mean_r']:.3f} |"
            )
    model = report["model_validation"]
    lines.extend(
        [
            "",
            "## Model validation",
            "",
            f"Mean accuracy: {model['mean_accuracy']:.1%}; median accuracy: {model['median_accuracy']:.1%}; mean Brier score: {model['mean_brier']:.4f}.",
            "",
            "## Decision",
            "",
            f"**{report['promotion_status']}** — {report['promotion_reason']}",
            "",
            "This report reuses the completed campaign's holdouts for diagnosis. Any candidate selected from it needs a fresh, untouched time period before it can qualify.",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_diagnostics(
        read_jsonl(RESULTS_PATH), read_jsonl(TRADES_PATH), read_jsonl(MODELS_PATH)
    )
    encoded = json.dumps(report, indent=2, default=lambda value: value.isoformat())
    DIAGNOSTICS_PATH.write_text(encoded + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(markdown_report(report), encoding="utf-8")

    repository = SQLiteRepository(str(JOURNAL_PATH))
    try:
        Journal(repository).record(
            Entity.AUDIT,
            datetime.now(UTC),
            {
                "event": "curated_history_loss_diagnostics_completed",
                "report": json.loads(encoded),
                "markdown_path": str(MARKDOWN_PATH.relative_to(ROOT)),
            },
            record_id=DIAGNOSTIC_ID,
        )
    finally:
        repository.close()
    print(
        f"Wrote {DIAGNOSTICS_PATH.relative_to(ROOT)} and {MARKDOWN_PATH.relative_to(ROOT)}; "
        f"journal record {DIAGNOSTIC_ID}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
