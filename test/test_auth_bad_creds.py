"""Test the app server with bad auth credentials"""
import unittest
from base64 import b64encode

from fastapi.testclient import TestClient

from app import app


class TestBadAuthCredentials(unittest.TestCase):
    """Test the app server with bad auth credentials"""

    def setUp(self):
        self.app = app
        self.client = TestClient(self.app)
        self.auth_headers = {
            "Authorization": "Basic " + b64encode(b"bad_user:bad_password").decode()
        }

    def test_failed_auth(self):
        """Test bad basic auth credentials"""
        response = self.client.get("/profiles", headers=self.auth_headers)
        print(response.content)
        self.assertEqual(response.status_code, 401)
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)
        self.assertEqual(response.json()["detail"], "Incorrect username or password")


if __name__ == "__main__":
    unittest.main()
