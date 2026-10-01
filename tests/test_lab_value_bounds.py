import unittest

from agent.intake import extract_lab_values


class LabValueBoundsTests(unittest.TestCase):
    def test_value_in_range_is_accepted(self):
        values = extract_lab_values("HbA1c: 6.1")
        self.assertEqual(values, {"HbA1c": 6.1})

    def test_out_of_range_value_is_dropped_and_reported(self):
        rejected: dict = {}
        values = extract_lab_values("HbA1c: 150", rejected=rejected)

        self.assertIsNone(values)
        self.assertEqual(
            rejected,
            {"HbA1c": {"value": 150.0, "min": 0, "max": 100}},
        )

    def test_extreme_but_real_blood_pressure_is_kept(self):
        values = extract_lab_values("ความดัน 280/150")

        self.assertEqual(values, {"SBP": 280.0, "DBP": 150.0})


if __name__ == "__main__":
    unittest.main()
