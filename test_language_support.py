import os
import unittest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from pydantic import ValidationError

from main import ChatRequest, normalize_language


class LanguageSupportTests(unittest.TestCase):
    def test_supported_languages(self):
        self.assertEqual(normalize_language("English"), "English")
        self.assertEqual(normalize_language("Odia"), "Odia")
        self.assertEqual(normalize_language("Hindi"), "Hindi")
        self.assertEqual(normalize_language("Sanskrit"), "Sanskrit")
        self.assertEqual(normalize_language("english"), "English")
        self.assertEqual(normalize_language("odia"), "Odia")
        self.assertEqual(normalize_language("hindi"), "Hindi")
        self.assertEqual(normalize_language("sanskrit"), "Sanskrit")

    def test_rejects_unsupported_languages(self):
        with self.assertRaises(ValueError):
            normalize_language("French")

    def test_chat_request_validates_language(self):
        request = ChatRequest(question="What is patent?", language="Odia")
        self.assertEqual(request.language, "Odia")

        request = ChatRequest(question="What is patent?", language="Hindi")
        self.assertEqual(request.language, "Hindi")

        with self.assertRaises(ValidationError):
            ChatRequest(question="What is patent?", language="French")


if __name__ == "__main__":
    unittest.main()
