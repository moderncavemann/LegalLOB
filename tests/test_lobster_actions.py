import pandas as pd

from genlob.data.lobster import MESSAGE_COLUMNS
from genlob.data.lobster_actions import derive_lobster_action_labels, lobster_actions_to_array


def test_lobster_execution_aggressor_side_is_opposite_resting_side():
    messages = pd.DataFrame(
        [
            [1.0, 4, 101, 10, 1000, -1],
            [2.0, 5, 102, 20, 1001, 1],
        ],
        columns=MESSAGE_COLUMNS,
    )

    labels = derive_lobster_action_labels(messages)

    assert labels.loc[0, "aggressor_side"] == 1
    assert labels.loc[1, "aggressor_side"] == -1
    assert labels.loc[0, "event_class"] == "execution"
    assert labels.loc[1, "event_class"] == "execution"


def test_lobster_submission_and_cancel_keep_resting_side_channels():
    messages = pd.DataFrame(
        [
            [1.0, 1, 101, 10, 1000, 1],
            [2.0, 2, 101, 5, 1000, 1],
            [3.0, 3, 101, 5, 1000, -1],
        ],
        columns=MESSAGE_COLUMNS,
    )

    labels = derive_lobster_action_labels(messages)
    actions = lobster_actions_to_array(labels, size_scale=10)

    assert labels.loc[0, "passive_side"] == 1
    assert labels.loc[1, "cancel_side"] == 1
    assert labels.loc[2, "cancel_side"] == -1
    assert actions.shape == (3, 5)
    assert actions[0, 1] == 1
    assert actions[1, 2] == 1
    assert actions[2, 2] == -1
