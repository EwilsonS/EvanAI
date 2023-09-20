"""Test wrap_urls() function."""
import unittest

from util.html import wrap_urls


class TestKeywordAnchors(unittest.TestCase):
    """Test wrap_urls() function."""

    def setUp(self):
        self.text = """
BMO web site
Open a checking account today!
<a href="https://bmo.com/global" title="https://bmo.com/global" rel="noopener" target="_blank">sample</a>
Plain URL: https://bmo.com/."""
        self.anchor_configs = [
            {
                "title": "BMO",
                "href": "https://bmo.com/",
                "regex": r"""\b(BMO's|BMO)\b""",
            },
            {
                "title": "C_A_TITLE",
                "href": "https://checking.com/u/",
                "regex": r"""\b(checking\s+accounts?|checking)\b""",
            },
        ]

    def test_multiple_use_cases(self):
        """Test wrap_urls() function with multiple use cases."""
        result = wrap_urls(self.text, self.anchor_configs)
        # noqa: E501
        expected = """
<a href="https://bmo.com/" title="BMO" rel="noopener" target="_blank">BMO</a> web site
Open a <a href="https://checking.com/u/" title="C_A_TITLE" rel="noopener" target="_blank">checking account</a> today!
<a href="https://bmo.com/global" title="https://bmo.com/global" rel="noopener" target="_blank">sample</a>
Plain URL: <a href="https://bmo.com/" title="https://bmo.com/" rel="noopener" target="_blank">https://bmo.com/</a>."""
        self.assertEqual(result, expected)

    def test_no_matches(self):
        """Test wrap_urls() function when no matches are found."""
        text = "No matches!"
        result = wrap_urls(text, self.anchor_configs)
        self.assertEqual(result, text)  # no changes should be made

    def test_keyword_replacement(self):
        """Test wrap_urls() function when keyword is replaced."""
        text = """
Open a checking account today!"""
        expected = """
Open a <a href="https://checking.com/u/" title="C_A_TITLE" rel="noopener" target="_blank">checking account</a> today!"""
        result = wrap_urls(text, self.anchor_configs)
        self.assertEqual(result, expected)  # no changes should be made

    def test_plain_urls_transform(self):
        """Test wrap_urls() function when plain URLs are transformed."""
        text = """
Plain URL: https://bmo.com/."""
        expected = """
Plain URL: <a href="https://bmo.com/" title="https://bmo.com/" rel="noopener" target="_blank">https://bmo.com/</a>."""
        result = wrap_urls(text, self.anchor_configs)
        self.assertEqual(result, expected)  # no changes should be made

    def test_anchor_tags_left_intact(self):
        """Test wrap_urls() function when anchor tags are left intact."""
        text = """
A <a href="https://bmo.com/global" title="https://bmo.com/global" rel="noopener" target="_blank">sample</a>."""
        expected = """
A <a href="https://bmo.com/global" title="https://bmo.com/global" rel="noopener" target="_blank">sample</a>."""
        result = wrap_urls(text, self.anchor_configs)
        self.assertEqual(result, expected)  # no changes should be made

    def test_empty_configs(self):
        """Test wrap_urls() function when configs is empty."""
        text = "Hello, World!"
        result = wrap_urls(text, [])
        self.assertEqual(result, text)  # no changes should be made


if __name__ == "__main__":
    unittest.main()
