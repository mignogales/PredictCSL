"""Version stamp for cached GiftEval metric outputs.

Bump this whenever a metric value cannot be corrected from the persisted
per-sample cache alone.  Stage 3 and its done-marker both reject older cells so
an ordinary pipeline rerun refreshes the affected forecasts and derived files.
"""

METRIC_SUITE_VER = 3
