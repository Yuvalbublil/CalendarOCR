from pathlib import Path

import pytest
import yaml

from calendar_ocr.calendar_ocr import _load_config, read_roi


class TestLoadConfig:
    def test_none_returns_empty_dict(self):
        assert _load_config(None) == {}

    def test_reads_real_roi_yaml(self):
        config = _load_config(Path("roi.yaml"))
        assert "roi" in config
        assert "time" in config

    def test_missing_file_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="not found"):
            _load_config(Path("nonexistent_file_xyz.yaml"))

    def test_invalid_yaml_not_mapping_raises_runtime_error(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- list item\n- another item\n")
        with pytest.raises(RuntimeError, match="mapping"):
            _load_config(bad_yaml)


class TestReadRoi:
    def test_valid_roi(self):
        result = read_roi({"roi": [9, 140, 426, 576]})
        assert result == (9, 140, 426, 576)

    def test_missing_key_returns_none(self):
        assert read_roi({}) is None

    def test_wrong_length_raises_value_error(self):
        with pytest.raises(ValueError):
            read_roi({"roi": [1, 2, 3]})

    def test_values_are_int(self):
        result = read_roi({"roi": [9.0, 140.5, 426.0, 576.0]})
        assert result is not None
        assert all(isinstance(v, int) for v in result)

    def test_non_list_returns_none(self):
        assert read_roi({"roi": "invalid"}) is None
