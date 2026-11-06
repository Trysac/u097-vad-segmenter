"""Behavior tests for the public API of vad_segmenter."""

from __future__ import annotations

import pytest

import vad_segmenter


def test_normalize_trims_and_lowercases() -> None:
    assert vad_segmenter.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        vad_segmenter.normalize(value)


def test_version_is_semver() -> None:
    parts = vad_segmenter.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
