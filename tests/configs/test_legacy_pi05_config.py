import pytest

from lerobot.configs.policies import migrate_legacy_null_rtc_config


def test_migrates_disabled_legacy_rtc_field():
    config = {"type": "pi05", "rtc_config": None, "chunk_size": 50}
    assert migrate_legacy_null_rtc_config(config) == {"type": "pi05", "chunk_size": 50}
    assert "rtc_config" in config


@pytest.mark.parametrize("rtc_config", [{"enabled": True}, {"enabled": False}])
def test_does_not_migrate_non_null_rtc_configuration(rtc_config):
    assert migrate_legacy_null_rtc_config({"type": "pi05", "rtc_config": rtc_config}) is None
