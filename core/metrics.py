# FILE: core/metrics.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Collect and expose counters, gauges, and timing metrics for runtime operations.
#   SCOPE: MetricsCollector interface, InMemoryMetricsCollector, OperationalMetricsRegistry facade
#   DEPENDS: none
#   LINKS: M-METRICS
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   OperationalMetricsRegistry - Facade for operational metric queries
#   MetricsCollector - Abstract metric collection interface
#   InMemoryMetricsCollector - Thread-safe in-memory metric storage
#   NoOpMetricsCollector - No-op stub for testing
#   DEFAULT_METRICS_COLLECTOR - Module-level default collector instance
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any

MetricTags = tuple[tuple[str, str], ...]


# START_CONTRACT: MetricSummary
#   PURPOSE: Represent a snapshot of collected counters, gauges, and timing series.
#   INPUTS: { counters: dict[str, dict[str, Any]] - Aggregated counter series, gauges: dict[str, dict[str, Any]] - Aggregated gauge series, timings: dict[str, dict[str, Any]] - Aggregated timing series }
#   OUTPUTS: { instance - Immutable metrics snapshot }
#   SIDE_EFFECTS: none
#   LINKS: M-METRICS
# END_CONTRACT: MetricSummary
@dataclass(frozen=True)
class MetricSummary:
    counters: dict[str, dict[str, Any]]
    gauges: dict[str, dict[str, Any]]
    timings: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": self.counters,
            "gauges": self.gauges,
            "timings": self.timings,
        }


# START_CONTRACT: MetricsCollector
#   PURPOSE: Define the abstract contract for counter, gauge, and timing collection backends.
#   INPUTS: {}
#   OUTPUTS: { instance - Abstract metric collection interface }
#   SIDE_EFFECTS: none
#   LINKS: M-METRICS
# END_CONTRACT: MetricsCollector
class MetricsCollector(ABC):
    @abstractmethod
    def increment(self, name: str, value: int = 1, *, tags: dict[str, str] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_gauge(
        self, name: str, value: int | float, *, tags: dict[str, str] | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe_timing(
        self, name: str, duration_ms: float, *, tags: dict[str, str] | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> MetricSummary:
        raise NotImplementedError


# START_CONTRACT: NoOpMetricsCollector
#   PURPOSE: Provide a metrics collector implementation that discards all observations.
#   INPUTS: {}
#   OUTPUTS: { instance - No-op metrics collector }
#   SIDE_EFFECTS: none
#   LINKS: M-METRICS
# END_CONTRACT: NoOpMetricsCollector
class NoOpMetricsCollector(MetricsCollector):
    def increment(self, name: str, value: int = 1, *, tags: dict[str, str] | None = None) -> None:
        return

    def set_gauge(
        self, name: str, value: int | float, *, tags: dict[str, str] | None = None
    ) -> None:
        return

    def observe_timing(
        self, name: str, duration_ms: float, *, tags: dict[str, str] | None = None
    ) -> None:
        return

    def snapshot(self) -> MetricSummary:
        return MetricSummary(counters={}, gauges={}, timings={})


# START_CONTRACT: InMemoryMetricsCollector
#   PURPOSE: Collect counters, gauges, and timings in process-local thread-safe memory.
#   INPUTS: {}
#   OUTPUTS: { instance - In-memory metrics collector }
#   SIDE_EFFECTS: Mutates in-memory metric series stored on the collector instance
#   LINKS: M-METRICS
# END_CONTRACT: InMemoryMetricsCollector
class InMemoryMetricsCollector(MetricsCollector):
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, MetricTags], int] = defaultdict(int)
        self._gauges: dict[tuple[str, MetricTags], float] = {}
        self._timings: dict[tuple[str, MetricTags], dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0}
        )

    def increment(self, name: str, value: int = 1, *, tags: dict[str, str] | None = None) -> None:
        key = (name, _normalize_tags(tags))
        with self._lock:
            self._counters[key] += value

    def set_gauge(
        self, name: str, value: int | float, *, tags: dict[str, str] | None = None
    ) -> None:
        key = (name, _normalize_tags(tags))
        with self._lock:
            self._gauges[key] = float(value)

    def observe_timing(
        self, name: str, duration_ms: float, *, tags: dict[str, str] | None = None
    ) -> None:
        key = (name, _normalize_tags(tags))
        with self._lock:
            bucket = self._timings[key]
            bucket["count"] += 1
            bucket["sum_ms"] += duration_ms
            bucket["max_ms"] = max(bucket["max_ms"], duration_ms)
            bucket["last_ms"] = duration_ms

    def snapshot(self) -> MetricSummary:
        with self._lock:
            counters = _group_metric_series(self._counters)
            gauges = _group_metric_series(self._gauges)
            timings = _group_timing_series(self._timings)
        return MetricSummary(counters=counters, gauges=gauges, timings=timings)


DEFAULT_METRICS_COLLECTOR = InMemoryMetricsCollector()


# START_CONTRACT: OperationalMetricsRegistry
#   PURPOSE: Expose higher-level readiness and execution summaries over a metrics collector.
#   INPUTS: { collector: MetricsCollector | None - Optional collector override used for metric aggregation }
#   OUTPUTS: { instance - Metrics facade for operational reporting }
#   SIDE_EFFECTS: none
#   LINKS: M-METRICS
# END_CONTRACT: OperationalMetricsRegistry
class OperationalMetricsRegistry:
    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self._collector = collector or InMemoryMetricsCollector()

    @property
    def collector(self) -> MetricsCollector:
        return self._collector

    def execution_summary(self) -> dict[str, Any]:
        snapshot = self._collector.snapshot().to_dict()
        counters = snapshot["counters"]
        gauges = snapshot["gauges"]
        return {
            "submitted": _counter_value(counters, "jobs.submitted"),
            "started": _counter_value(counters, "jobs.started"),
            "completed": _counter_value(counters, "jobs.completed"),
            "failed": _counter_value(counters, "jobs.failed"),
            "timeout": _counter_value(counters, "jobs.timeout"),
            "cancelled": _counter_value(counters, "jobs.cancelled"),
            "queue_depth": {
                "current": _gauge_value(gauges, "jobs.queue.depth.current"),
                "peak": _gauge_value(gauges, "jobs.queue.depth.peak"),
            },
        }

    def model_summary(self) -> dict[str, Any]:
        snapshot = self._collector.snapshot().to_dict()
        counters = snapshot["counters"]
        timings = snapshot["timings"]
        return {
            "cache": {
                "hit": _split_counter_by_tag(counters, "models.cache.hit", "backend"),
                "miss": _split_counter_by_tag(counters, "models.cache.miss", "backend"),
            },
            "load": {
                "failures": _split_counter_by_tag(counters, "models.load.failed", "backend"),
                "duration_ms": _split_timing_by_tag(timings, "models.load.duration_ms", "backend"),
            },
        }

    def readiness_summary(self) -> dict[str, Any]:
        return {
            "execution": self.execution_summary(),
            "models": self.model_summary(),
        }


def _normalize_tags(tags: dict[str, str] | None) -> MetricTags:
    if not tags:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in tags.items()))


def _group_metric_series(
    series: dict[tuple[str, MetricTags], int | float],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for (name, tags), value in series.items():
        grouped.setdefault(name, {"total": 0, "series": []})
        grouped[name]["total"] += value
        grouped[name]["series"].append({"tags": dict(tags), "value": value})
    for item in grouped.values():
        item["series"].sort(key=lambda entry: sorted(entry["tags"].items()))
    return grouped


def _group_timing_series(
    series: dict[tuple[str, MetricTags], dict[str, float]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for (name, tags), value in series.items():
        grouped.setdefault(name, {"count": 0, "sum_ms": 0.0, "max_ms": 0.0, "series": []})
        grouped[name]["count"] += int(value["count"])
        grouped[name]["sum_ms"] += value["sum_ms"]
        grouped[name]["max_ms"] = max(grouped[name]["max_ms"], value["max_ms"])
        grouped[name]["series"].append(
            {
                "tags": dict(tags),
                "count": int(value["count"]),
                "sum_ms": round(value["sum_ms"], 3),
                "max_ms": round(value["max_ms"], 3),
                "avg_ms": round(value["sum_ms"] / value["count"], 3) if value["count"] else 0.0,
                "last_ms": round(value["last_ms"], 3),
            }
        )
    for item in grouped.values():
        item["sum_ms"] = round(item["sum_ms"], 3)
        item["max_ms"] = round(item["max_ms"], 3)
        item["avg_ms"] = round(item["sum_ms"] / item["count"], 3) if item["count"] else 0.0
        item["series"].sort(key=lambda entry: sorted(entry["tags"].items()))
    return grouped


def _counter_value(counters: dict[str, dict[str, Any]], name: str) -> int:
    bucket = counters.get(name)
    return 0 if bucket is None else int(bucket["total"])


def _gauge_value(gauges: dict[str, dict[str, Any]], name: str) -> int:
    bucket = gauges.get(name)
    if bucket is None:
        return 0
    return int(bucket["total"])


def _split_counter_by_tag(
    counters: dict[str, dict[str, Any]], name: str, tag_name: str
) -> dict[str, int]:
    bucket = counters.get(name)
    if bucket is None:
        return {}
    values: dict[str, int] = {}
    for item in bucket["series"]:
        values[item["tags"].get(tag_name, "unknown")] = int(item["value"])
    return values


def _split_timing_by_tag(
    timings: dict[str, dict[str, Any]], name: str, tag_name: str
) -> dict[str, dict[str, Any]]:
    bucket = timings.get(name)
    if bucket is None:
        return {}
    values: dict[str, dict[str, Any]] = {}
    for item in bucket["series"]:
        values[item["tags"].get(tag_name, "unknown")] = {
            "count": item["count"],
            "avg_ms": item["avg_ms"],
            "max_ms": item["max_ms"],
            "last_ms": item["last_ms"],
        }
    return values


__all__ = [
    "DEFAULT_METRICS_COLLECTOR",
    "InMemoryMetricsCollector",
    "MetricsCollector",
    "NoOpMetricsCollector",
    "OperationalMetricsRegistry",
]
