from pathlib import Path

from scripts.audit_external_baselines import required_path_checks, scan_device_markers


def test_required_path_checks(tmp_path: Path):
    (tmp_path / "README.md").write_text("baseline")
    checks = required_path_checks(tmp_path, ("README.md", "missing.py"))
    assert checks == {"README.md": True, "missing.py": False}


def test_device_marker_scan_detects_cuda_only_code(tmp_path: Path):
    (tmp_path / "trainer.py").write_text(
        'device = "cuda" if torch.cuda.is_available() else "cpu"\n'
        'with torch.amp.autocast(device_type=device):\n'
        '    pass\n'
    )
    markers = scan_device_markers(tmp_path)
    assert markers["cuda_only_selection"] == 1
    assert markers["cuda_mentions"] >= 2
    assert markers["mps_mentions"] == 0
    assert markers["amp_mentions"] == 1
