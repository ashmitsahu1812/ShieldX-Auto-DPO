import json
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

def generate_synthetic_case(topic: str) -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    prompt = f"""Generate a realistic JSON object representing a GitHub pull request with a hidden software defect related to: {topic}.
    
Expected JSON format EXACTLY:
{{
  "task_id": "unique_synthetic_id",
  "title": "Clear PR title",
  "description": "Short description of PR",
  "files_changed": [
    {{
      "filename": "path/to/file.ext",
      "status": "modified",
      "additions": 5,
      "deletions": 2,
      "patch": "@@ -1,5 +1,5 @@\\n unified diff format with + and - markers"
    }}
  ],
  "ground_truth_bugs": [
    {{
      "file": "path/to/file.ext",
      "keyword": "short_bug_keyword_like_sql_injection"
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    start = content.find("{")
    end = content.rfind("}")
    return json.loads(content[start:end+1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="Race condition in dictionary update", help="Topic for the bug")
    args = parser.add_argument()
    
    # Just an example script; disabled execution unless token is valid.
    print(f"To generate cases, ensure your API keys are set. Topic intended: {args.topic}")
