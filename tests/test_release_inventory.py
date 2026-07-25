"""Fast checks for the public release inventory and frozen artifacts."""

from verify_results import (
    check_inventory,
    check_model_construction,
    check_split_hashes,
)


def test_release_inventory() -> None:
    check_inventory()


def test_frozen_split_hashes() -> None:
    check_split_hashes()


def test_released_model_construction() -> None:
    check_model_construction()
