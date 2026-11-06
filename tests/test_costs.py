"""Transaction cost tests."""

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs, apply_slippage_to_premiums


def test_zero_costs():
    c = CostConfig()
    breakdown = compute_transaction_costs(100, 150, 10, 5, c)
    assert breakdown["total_cost"] == 0.0


def test_brokerage_and_stt_reduce_net():
    c = CostConfig(brokerage_per_lot=20, stt_pct=0.001)
    # 5 lots, lot_size 10 → qty 50
    # brokerage = 20 * 5 * 2 = 200
    # STT on exit turnover = 0.001 * 150 * 50 = 7.5
    breakdown = compute_transaction_costs(100, 150, 10, 5, c)
    assert breakdown["brokerage"] == 200
    assert abs(breakdown["stt"] - 7.5) < 1e-6
    assert breakdown["total_cost"] > 0


def test_slippage_adjusts_premiums():
    c = CostConfig(entry_slippage=2, exit_slippage=3)
    e, x = apply_slippage_to_premiums(100, 150, c)
    assert e == 102
    assert x == 147


def test_cost_model_scales_brokerage_and_slippage_by_legs():
    c = CostConfig(brokerage_per_lot=20, entry_slippage=1, exit_slippage=1)
    breakdown = compute_transaction_costs(10, 12, 10, 2, c, n_legs=2)
    assert breakdown["brokerage"] == 160
    assert breakdown["slippage"] == 80
