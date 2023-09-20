""" ChatGPT test for sending sample user prompts
and recording resulting conversations for future analysis
"""
import csv
import os

import openai
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# set API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    not os.environ.get("TEST_SAMPLES"),
    reason="API key not set",
)  # bash: `TEST_SAMPLES=true PYTHONPATH=. pytest`
def test_talk_to_chat():
    """Get ChatGPT answers to sample user prompts"""
    samples = os.listdir("test/samples")
    for sample in samples:
        if "csv" in sample:
            with open(
                f"test/samples/{sample}",
                newline="",
                encoding="utf-8",
                errors="ignore",
            ) as file:
                reader = csv.reader(file, delimiter=",")
                system = next(reader)[0]
                messages = []
                messages.append({"role": "system", "content": system})
                if os.path.exists(f"test/test_results/{sample}"):
                    os.remove(f"test/test_results/{sample}")
                with open(
                    f"test/test_results/{sample}", "a", encoding="utf-8"
                ) as result:
                    result.write(f'"{system}"\n')
                    while (user_message := next(reader, None)) is not None:
                        messages.append({"role": "user", "content": user_message[0]})
                        result.write(f'"{user_message[0]}"\n')
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            n=1,
                            temperature=0.3,
                        )
                        ai_message = response.choices[0].message
                        messages.append(ai_message)
                        formatted = ai_message.content.replace("\n", "\\n")
                        result.write(f'"{formatted}"\n')
