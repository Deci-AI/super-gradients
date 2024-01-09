import os
import unittest
from .breaking_changes_detection import analyze_breaking_changes


class BreakingChangeTest(unittest.TestCase):
    def test_detect_breaking_change(self):

        source_branch = os.getenv("BRANCH_MERGED_INTO", "master")
        breaking_changes_list = analyze_breaking_changes(verbose=True, source_branch=source_branch)
        self.assertEqual(len(breaking_changes_list), 0, f"{len(breaking_changes_list)} breaking changes detected")


if __name__ == "__main__":
    unittest.main()
