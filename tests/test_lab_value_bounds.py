import unittest

from agent.intake import extract_lab_values


class LabValueBoundsTests(unittest.TestCase):
    def test_plausible_value_is_accepted(self):
        values = extract_lab_values("HbA1c: 6.1")
        self.assertEqual(values, {"HbA1c": 6.1})

    def test_implausible_value_is_dropped_and_reported(self):
        rejected: dict = {}
        values = extract_lab_values("HbA1c: 2", rejected=rejected)

        self.assertIsNone(values)
        self.assertEqual(
            rejected,
            {"HbA1c": {"value": 2.0, "min": 3, "max": 20}},
        )

    def test_implausible_blood_pressure_pair_reports_only_the_bad_side(self):
        rejected: dict = {}
        values = extract_lab_values("ความดัน 999/80", rejected=rejected)

        self.assertEqual(values, {"DBP": 80.0})
        self.assertEqual(
            rejected,
            {"SBP": {"value": 999.0, "min": 50, "max": 260}},
        )


if __name__ == "__main__":
    unittest.main()
