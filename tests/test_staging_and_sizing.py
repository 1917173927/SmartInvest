from __future__ import annotations

from datetime import date

from stock_analysis.decision import (
    ValuationRange,
    compute_staging_plan,
)
from stock_analysis.indicators import PriceZone


def test_compute_staging_plan_basic() -> None:
    vr = ValuationRange(
        available=True,
        currency="CNY",
        fair_low=50.0,
        fair_high=60.0,
        buy_low=45.0,
        buy_high=52.0,
        method="test",
    )
    price_zones = [
        PriceZone(
            kind="support",
            low=48.0,
            high=49.0,
            center=48.5,
            distance=-0.03,
            strength=0.8,
            touches=5,
            last_touch=date(2026, 1, 1),
        ),
        PriceZone(
            kind="support",
            low=44.0,
            high=45.0,
            center=44.5,
            distance=-0.11,
            strength=0.7,
            touches=3,
            last_touch=date(2026, 1, 1),
        ),
    ]

    plan = compute_staging_plan(
        current_price=50.0,
        valuation_range=vr,
        price_zones=price_zones,
        role="core",
        total_capital=100000.0,
        target_position=0.20,
    )

    assert plan.available
    assert plan.total_target_weight == 0.20
    assert len(plan.tiers) == 3
    assert plan.tiers[0].shares > 0
    assert plan.tiers[1].shares > 0
    assert plan.tiers[2].shares > 0
    # Shares are in multiples of 100
    assert all(t.shares % 100 == 0 for t in plan.tiers)
    # Monotonic price levels
    assert plan.tiers[0].target_price >= plan.tiers[1].target_price >= plan.tiers[2].target_price
    assert plan.invalidation_price is not None
    assert plan.invalidation_price < plan.tiers[2].target_price


def test_staging_plan_never_exceeds_remaining_target_budget() -> None:
    vr = ValuationRange(available=False, method="test")

    plan = compute_staging_plan(
        current_price=55.82,
        valuation_range=vr,
        price_zones=[],
        role="core",
        total_capital=51546.80,
        target_position=0.20,
    )

    assert plan.total_capital <= 51546.80 * 0.20
    assert sum(tier.allocated_amount for tier in plan.tiers) == plan.total_capital


def test_staging_plan_has_no_new_orders_when_existing_position_exceeds_target() -> None:
    plan = compute_staging_plan(
        current_price=55.82,
        valuation_range=ValuationRange(available=False, method="test"),
        price_zones=[],
        total_capital=51546.80,
        target_position=0.20,
        existing_position_value=300 * 57.10,
    )

    assert plan.total_shares == 0
    assert plan.total_capital == 0
    assert all(tier.shares == 0 for tier in plan.tiers)
