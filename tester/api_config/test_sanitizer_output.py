from __future__ import annotations

import unittest

from tester.api_config.sanitizer_output import analyze_sanitizer_output


class SanitizerOutputTest(unittest.TestCase):
    def analyze(self, output, returncode=86):
        return analyze_sanitizer_output(output, returncode, 86)

    def test_cuda_version_diagnostic_is_completely_silent(self):
        result = self.analyze(
            "========= CUDA API Error: cudaErrorInvalidValue (error 1)\n"
            "========= cudaVersion argument (12090) exceeds the driver version (12080)\n"
            "=========     Saved host backtrace up to driver entry point\n"
            "========= ERROR SUMMARY: 1 error\n"
        )
        self.assertEqual(result.output, "")
        self.assertTrue(result.only_ignored_diagnostics)

    def test_cu_get_proc_address_diagnostic_is_completely_silent(self):
        result = self.analyze(
            "========= CUDA API Error: CUDA_ERROR_INVALID_VALUE\n"
            "=========     cuGetProcAddress_v2\n"
            "========= ERROR SUMMARY: 1 error\n"
        )
        self.assertEqual(result.output, "")
        self.assertTrue(result.only_ignored_diagnostics)

    def test_real_error_is_preserved_when_mixed_with_ignored_error(self):
        result = self.analyze(
            "child output\n"
            "========= CUDA API Error: cudaErrorInvalidValue\n"
            "========= cudaVersion argument (13000) exceeds the driver version (12080)\n"
            "========= Program hit cudaErrorIllegalAddress\n"
            "=========     at kernel.cu:10\n"
            "========= ERROR SUMMARY: 2 errors\n"
        )
        self.assertEqual(
            result.output,
            "child output\n"
            "========= Program hit cudaErrorIllegalAddress\n"
            "=========     at kernel.cu:10\n"
            "========= ERROR SUMMARY: 2 errors",
        )
        self.assertFalse(result.only_ignored_diagnostics)

    def test_non_sanitizer_exit_is_not_filtered(self):
        output = (
            "========= CUDA API Error: CUDA_ERROR_INVALID_VALUE\n========= cuGetProcAddress_v2\n"
        )
        result = self.analyze(output, returncode=2)
        self.assertEqual(result.output, output)
        self.assertFalse(result.only_ignored_diagnostics)


if __name__ == "__main__":
    unittest.main()
