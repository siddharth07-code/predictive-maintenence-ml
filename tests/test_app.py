from contextlib import chdir
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]


class ModelComparisonTest(unittest.TestCase):
    def assert_comparison_available(self, app):
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe), 1)
        table = app.dataframe[0].value
        self.assertEqual(len(table), 4)
        self.assertEqual(table["Model"].tolist(), EXPECTED_MODELS)

        downloads = app.get("download_button")
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0].proto.label, "Download comparison CSV")
        self.assertFalse(downloads[0].proto.disabled)
        self.assertTrue(downloads[0].proto.url)

    def test_torque_change_requires_fresh_comparison(self):
        with chdir(REPO_ROOT):
            app = AppTest.from_file(
                str(REPO_ROOT / "app.py"), default_timeout=30
            ).run()
            self.assertEqual(len(app.exception), 0)

            app.selectbox(key="example").select(
                "Recorded failure (ID 51)"
            ).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.number_input(key="torque").value, 4.6)
            self.assertEqual(app.selectbox(key="machine_type").value, "L")

            compare_buttons = [
                button for button in app.button if button.label == "Compare models"
            ]
            self.assertEqual(len(compare_buttons), 1)
            compare_buttons[0].click().run()
            self.assert_comparison_available(app)

            app.number_input(key="torque").set_value(5.6).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.number_input(key="torque").value, 5.6)
            self.assertEqual(len(app.dataframe), 0)
            self.assertEqual(len(app.get("download_button")), 0)

            compare_buttons = [
                button for button in app.button if button.label == "Compare models"
            ]
            self.assertEqual(len(compare_buttons), 1)
            compare_buttons[0].click().run()
            self.assert_comparison_available(app)


if __name__ == "__main__":
    unittest.main()
