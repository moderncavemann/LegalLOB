import numpy as np
import pandas as pd

from genlob.constraints import (
    check_lobster_orderbook_invariants,
    project_lobster_messages,
    validate_lobster_messages,
)
from genlob.data.lobster import (
    LOBSTER_ASK_DUMMY_PRICE,
    LOBSTER_BID_DUMMY_PRICE,
    MESSAGE_COLUMNS,
)


def test_lobster_orderbook_invariant_report_ignores_dummy_levels():
    orderbook = np.asarray(
        [
            [1_000_100, 10, 1_000_000, 20, LOBSTER_ASK_DUMMY_PRICE, 0, LOBSTER_BID_DUMMY_PRICE, 0],
            [1_000_100, 10, 1_000_000, 20, 1_000_200, 5, 999_900, 4],
        ],
        dtype=np.int64,
    )

    report = check_lobster_orderbook_invariants(orderbook, depth=2)

    assert report.ok
    assert report.crossed_book_rate == 0.0


def test_lobster_orderbook_invariant_report_finds_crossed_and_ladder_violations():
    orderbook = np.asarray(
        [
            [1_000_000, 10, 1_000_100, 20, 999_900, 5, 1_000_200, 4],
        ],
        dtype=np.int64,
    )

    report = check_lobster_orderbook_invariants(orderbook, depth=2)

    assert report.crossed_book_rate == 1.0
    assert report.ask_ladder_violation_rate == 1.0
    assert report.bid_ladder_violation_rate == 1.0


def test_lobster_message_grammar_projection_fixes_field_domain():
    raw = pd.DataFrame(
        [[0.0, 9, -5, -10, -123, 0], [1.0, 1, 1, 10, 100, 1]],
        columns=MESSAGE_COLUMNS,
    )

    before = validate_lobster_messages(raw)
    projected = project_lobster_messages(raw, tick_size=5)
    after = validate_lobster_messages(projected)

    assert before.total_violations > 0
    assert after.ok
    assert projected.loc[0, "EventType"] == 7
    assert projected.loc[0, "OrderID"] == 1
    assert projected.loc[0, "Size"] == 1
    assert projected.loc[0, "Price"] == 5
    assert projected.loc[0, "Direction"] == 1
