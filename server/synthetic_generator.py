import json
import random
import logging
from typing import List, Dict, Any
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# Rotating topics for diverse dataset growth
BUG_TOPICS = [
    "SQL Injection in a user search function",
    "Race condition in a shared dictionary update without locks",
    "Mutable default argument in a Python data pipeline",
    "Off-by-one error in a list processing loop",
    "Unchecked null response from a third-party API in a React component",
    "Sensitive developer token leaked via hardcoded YAML string",
    "Inefficient O(N^2) nested loop in a high-frequency utility",
    "Broken authentication logic allowing empty password bypass",
    "Incorrect CSS specificity causing invisible elements",
    "React useEffect missing dependency causing infinite loop"
]

class SyntheticGenerator:
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def generate_case(self, topic: str = None) -> Dict[str, Any]:
        """Generate a single high-quality synthetic PR case."""
        if not topic:
            topic = random.choice(BUG_TOPICS)
            
        logger.info(f"Generating synthetic case for topic: {topic}")
        
        prompt = f"""Generate a high-quality, realistic JSON simulation case for an AI code review game.
The case must revolve around: {topic}.

JSON SCHEMA:
{{
  "pr_id": "SYNTH-XXXX",
  "title": "Short title describing the PR context",
  "description": "2-3 sentences of PR description",
  "files_changed": [
    {{
      "filename": "path/to/buggy_file.py",
      "diff": "@@ -1,5 +1,5 @@\\n unified diff patch containing the bug"
    }}
  ],
  "ground_truth_bugs": [
    {{
      "type": "logic|security|style",
      "file": "path/to/buggy_file.py",
      "keyword": "one-word-keyword-for-the-bug"
    }}
  ],
  "expected_action": "request_changes",
  "language": "python|javascript|go|css",
  "framework": "general|django|react"
}}

IMPORTANT:
1. The 'diff' MUST show the bug clearly using + markers.
2. The code must look professional and idiomatic.
3. Return ONLY the JSON object. No prose.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1500
            )
            content = response.choices[0].message.content.strip()
            # Robust JSON extraction
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("Model failed to output valid JSON")
            
            case_data = json.loads(content[start:end+1])
            return case_data
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")
            raise e
