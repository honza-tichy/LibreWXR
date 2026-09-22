# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Settings must load from the environment the way the docs spell it.

Regression 2026-09-22: radar_priority_groups was declared list[str].
pydantic-settings decodes a list-typed field from the environment as
JSON *inside the settings source*, before any field_validator runs, so
the documented comma-separated spelling raised SettingsError at import
and crash-looped the container.  Every list-shaped knob here is
documented comma-separated, so each one needs a str field and an
explicit split.
"""

import pytest


@pytest.mark.parametrize(
    "env_name,field,value,expected_split",
    [
        (
            "LIBREWXR_RADAR_PRIORITY_GROUPS",
            "radar_priority_groups",
            "US,CANADA,EUROPE",
            ["US", "CANADA", "EUROPE"],
        ),
        (
            "LIBREWXR_ENABLED_REGIONS",
            "enabled_regions",
            "CONUS,EUROPE",
            ["CONUS", "EUROPE"],
        ),
    ],
)
def test_comma_separated_env_vars_load(
    monkeypatch, env_name, field, value, expected_split
):
    from librewxr.config import Settings

    monkeypatch.setenv(env_name, value)
    settings = Settings()

    assert getattr(settings, field) == value
    assert [p.strip() for p in getattr(settings, field).split(",")] == expected_split


def test_empty_priority_groups_loads():
    """The documented way to opt out of wave splitting."""
    import os
    from librewxr.config import Settings

    os.environ["LIBREWXR_RADAR_PRIORITY_GROUPS"] = ""
    try:
        assert Settings().radar_priority_groups == ""
    finally:
        del os.environ["LIBREWXR_RADAR_PRIORITY_GROUPS"]
