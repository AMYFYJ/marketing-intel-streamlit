from __future__ import annotations

import pandas as pd

from data_sources.synthetic_media_mix import (
    DEFAULT_SYNTHETIC_ROWS,
    build_channel_benchmarks,
    generate_synthetic_media_mix,
    validate_media_mix_frame,
)


def test_build_channel_benchmarks_returns_platform_mix() -> None:
    benchmarks = build_channel_benchmarks()

    assert set(["platform", "spend_share", "ctr", "cvr", "cpm", "aov", "roas"]).issubset(benchmarks.columns)
    assert round(float(benchmarks["spend_share"].sum()), 6) == 1.0
    assert benchmarks["ctr"].gt(0).all()
    assert benchmarks["cpm"].gt(0).all()


def test_generate_synthetic_media_mix_is_deterministic() -> None:
    left = generate_synthetic_media_mix(rows=1_000, seed=99)
    right = generate_synthetic_media_mix(rows=1_000, seed=99)

    pd.testing.assert_frame_equal(left, right)


def test_generate_synthetic_media_mix_constraints() -> None:
    frame = generate_synthetic_media_mix(rows=5_000, seed=123)
    errors = validate_media_mix_frame(frame)

    assert len(frame) == 5_000
    assert errors == []
    assert frame["roas"].ge(0).all()
    assert frame["cpa"].ge(0).all()
    assert frame["profit"].equals(frame["revenue"] - frame["spend"])


def test_validate_media_mix_frame_catches_invalid_constraints() -> None:
    frame = generate_synthetic_media_mix(rows=10, seed=3)
    frame.loc[0, "clicks"] = frame.loc[0, "impressions"] + 1

    errors = validate_media_mix_frame(frame)

    assert any("clicks cannot exceed impressions" in error for error in errors)


def test_default_synthetic_scale_constant() -> None:
    assert DEFAULT_SYNTHETIC_ROWS == 250_000
