"""Test the Flask app"""
import json
import unittest
from base64 import b64encode
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app


class FastAPIAppTestCase(unittest.TestCase):
    """Test the Flask app"""

    def setUp(self):
        self.app = app
        self.client = TestClient(self.app)
        self.auth_headers = {"Authorization": "Basic " + b64encode(b"ci:cd").decode()}

    def test_get_profiles(self):
        """Test the profiles route"""
        response = self.client.get("/profiles", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)
        self.assertTrue(len(response.json()) > 0)

    @patch("openai.ChatCompletion.create")
    def test_generate(self, mock_chat_completion):
        """Test the generate route"""
        control_value = "test message"
        mock_chat_completion.return_value = {
            "choices": [{"message": {"content": control_value}}]
        }

        response = self.client.post(
            "/generate",
            content=json.dumps(
                {
                    "message": "hello",
                    "history": [{"role": "user", "content": "hello world"}],
                }
            ),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic " + b64encode(b"ci:cd").decode(),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json())
        self.assertEqual(response.json()["message"], control_value)


if __name__ == "__main__":
    unittest.main()
