from __future__ import annotations

from pathlib import Path


def test_readme_documents_deployment_and_sources() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Streamlit Community Cloud" in readme
    assert "Digital Advertising Campaign Performance Dataset" in readme
    assert "Meta Ad Library" in readme
    assert "TikTok Creative Center" in readme


def test_secrets_template_exists() -> None:
    template = Path(".streamlit/secrets.toml.example")

    assert template.exists()
    text = template.read_text(encoding="utf-8")
    assert "YOUTUBE_API_KEY" in text
    assert "META_ACCESS_TOKEN" in text
