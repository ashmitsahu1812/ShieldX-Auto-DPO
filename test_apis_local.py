import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")

def test_hf():
    print("--- Testing Hugging Face Tier ---")
    if not HF_TOKEN:
        print("HF_TOKEN missing!")
        return
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"HF Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"HF Failed: {e}")

def test_pollinations():
    print("\n--- Testing Pollinations Tier ---")
    client = OpenAI(base_url="https://text.pollinations.ai/openai", api_key="not-needed")
    # Trying different models to find the right one
    models = ["openai", "qwen", "llama"]
    for m in models:
        try:
            print(f"Trying Pollinations model: {m}...")
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10,
                timeout=10
            )
            print(f"Pollinations {m} Success: {response.choices[0].message.content}")
            return
        except Exception as e:
            print(f"Pollinations {m} Failed: {e}")

if __name__ == "__main__":
    test_hf()
    test_pollinations()
