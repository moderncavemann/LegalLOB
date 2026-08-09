from scripts.hash_artifacts import _paths_from_markdown


def test_paths_from_markdown_includes_paper_figures(tmp_path) -> None:
    index = tmp_path / "ARTIFACT_INDEX.md"
    index.write_text(
        "| Figure | `paper_figures/figure.png` |\n"
        "```text\nignored/inside_code.png\n```\n"
    )

    assert _paths_from_markdown(index) == ["paper_figures/figure.png"]
