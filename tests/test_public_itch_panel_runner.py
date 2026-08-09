import json

import pytest

from scripts.run_public_itch_panel_experiments import validate_panel_audit


def test_panel_runner_requires_audited_datasets(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "all_schema_checks_pass": True,
                "datasets": [
                    {"dataset": "PANEL_AAPL_1s", "status": "pass"},
                    {"dataset": "PANEL_GOOG_1s", "status": "pass_with_warnings"},
                ],
            }
        )
    )

    payload = validate_panel_audit(audit, ["PANEL_AAPL_1s", "PANEL_GOOG_1s"])
    assert payload["all_schema_checks_pass"] is True
    with pytest.raises(ValueError, match="incomplete"):
        validate_panel_audit(audit, ["PANEL_AMZN_1s"])


def test_panel_runner_rejects_failed_global_audit(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"all_schema_checks_pass": False, "datasets": []}))

    with pytest.raises(ValueError, match="has not passed"):
        validate_panel_audit(audit, ["PANEL_AAPL_1s"])
