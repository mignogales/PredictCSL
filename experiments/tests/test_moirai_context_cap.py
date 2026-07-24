import unittest

from experiments import test_window_ablation_gifteval_v5 as ablation


class MoiraiContextCapTest(unittest.TestCase):
    def test_full_native_cap_reserves_the_forecast_horizon(self):
        self.assertEqual(
            ablation._full_native_context_cap("moirai", 480, 104_640),
            7_712,
        )

    def test_full_native_cap_does_not_expand_short_context(self):
        self.assertEqual(
            ablation._full_native_context_cap("moirai", 480, 4_096),
            4_096,
        )

    def test_cap_accounts_for_patch_padding(self):
        self.assertEqual(ablation._moirai_max_context(1), 8_176)

    def test_other_uncapped_family_still_uses_available_context(self):
        self.assertEqual(
            ablation._full_native_context_cap(
                "context_parroting", 480, 104_640
            ),
            104_640,
        )


if __name__ == "__main__":
    unittest.main()
