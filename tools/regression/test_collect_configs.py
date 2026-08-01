from __future__ import annotations

from tester.paddle_to_torch.rules import Code, ConvertResult
from tools.regression import collect_configs


def test_is_accuracy_supported_accepts_successful_convert_result(monkeypatch):
    class Converter:
        def convert(self, api_name):
            return ConvertResult.success(api_name, Code(core=("result = None",)))

    monkeypatch.setattr(collect_configs, "get_converter", lambda: Converter())

    assert collect_configs.is_accuracy_supported("paddle.ones", {}) is True


def test_is_accuracy_supported_rejects_error_convert_result(monkeypatch):
    class Converter:
        def convert(self, api_name):
            return ConvertResult.error(api_name, "unsupported")

    monkeypatch.setattr(collect_configs, "get_converter", lambda: Converter())

    assert collect_configs.is_accuracy_supported("paddle.missing", {}) is False
