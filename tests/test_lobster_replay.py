import numpy as np
import pandas as pd

from genlob.data.lobster import (
    LOBSTER_ASK_DUMMY_PRICE,
    LOBSTER_BID_DUMMY_PRICE,
    MESSAGE_COLUMNS,
    read_message_csv,
    read_orderbook_csv,
)
from genlob.matcher import compare_replay_to_orderbook, replay_lobster_messages


def _messages():
    rows = [
        [34200.001, 1, 1, 100, 1_000_000, 1],
        [34200.002, 1, 2, 50, 1_000_100, -1],
        [34200.003, 1, 3, 70, 999_900, 1],
        [34200.004, 2, 1, 40, 1_000_000, 1],
        [34200.005, 4, 2, 20, 1_000_100, -1],
        [34200.006, 5, 0, 10, 1_000_100, -1],
        [34200.007, 3, 1, 60, 1_000_000, 1],
        [34200.008, 6, 0, 10, 1_000_050, 0],
        [34200.009, 7, 0, 0, 0, 0],
    ]
    return pd.DataFrame(rows, columns=MESSAGE_COLUMNS)


def _expected_orderbook():
    ask_dummy = LOBSTER_ASK_DUMMY_PRICE
    bid_dummy = LOBSTER_BID_DUMMY_PRICE
    rows = [
        [ask_dummy, 0, 1_000_000, 100, ask_dummy, 0, bid_dummy, 0],
        [1_000_100, 50, 1_000_000, 100, ask_dummy, 0, bid_dummy, 0],
        [1_000_100, 50, 1_000_000, 100, ask_dummy, 0, 999_900, 70],
        [1_000_100, 50, 1_000_000, 60, ask_dummy, 0, 999_900, 70],
        [1_000_100, 30, 1_000_000, 60, ask_dummy, 0, 999_900, 70],
        [1_000_100, 30, 1_000_000, 60, ask_dummy, 0, 999_900, 70],
        [1_000_100, 30, 999_900, 70, ask_dummy, 0, bid_dummy, 0],
        [1_000_100, 30, 999_900, 70, ask_dummy, 0, bid_dummy, 0],
        [1_000_100, 30, 999_900, 70, ask_dummy, 0, bid_dummy, 0],
    ]
    return np.asarray(rows, dtype=np.int64)


def test_replay_lobster_messages_matches_expected_orderbook():
    result = replay_lobster_messages(_messages(), depth=2)

    assert result.ok
    np.testing.assert_array_equal(result.states, _expected_orderbook())
    assert result.counts["event_type_1"] == 3
    assert result.counts["event_type_5"] == 1
    assert result.counts["event_type_7"] == 1


def test_compare_replay_to_orderbook_reports_exact_match():
    comparison = compare_replay_to_orderbook(_messages(), _expected_orderbook(), depth=2)

    assert comparison.exact
    assert comparison.match_rate == 1.0
    assert comparison.first_mismatches == []


def test_compare_replay_to_orderbook_reports_mismatch_context():
    expected = _expected_orderbook()
    expected[3, 3] = 999

    comparison = compare_replay_to_orderbook(_messages(), expected, depth=2, max_mismatches=1)

    assert not comparison.exact
    assert comparison.matched_rows == len(expected) - 1
    assert comparison.first_mismatches[0]["row"] == 3
    assert comparison.first_mismatches[0]["columns"] == [3]


def test_lobster_csv_readers_accept_headerless_files(tmp_path):
    message_path = tmp_path / "message.csv"
    orderbook_path = tmp_path / "orderbook.csv"

    _messages().to_csv(message_path, header=False, index=False)
    pd.DataFrame(_expected_orderbook()).to_csv(orderbook_path, header=False, index=False)

    messages = read_message_csv(message_path)
    orderbook = read_orderbook_csv(orderbook_path, depth=2)

    assert list(messages.columns) == MESSAGE_COLUMNS
    assert messages["Price"].dtype == np.int64
    assert orderbook.shape == (9, 8)
