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
        """Generate a single high-quality synthetic PR case with robust fallback."""
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
      "diff": "@@ -1,5 +1,5 @@\\n unified diff patch"
    }}
  ],
  "ground_truth_bugs": [ {{ "type": "logic", "file": "path/to/buggy_file.py", "keyword": "bug" }} ],
  "expected_action": "request_changes",
  "language": "python",
  "framework": "general"
}}
IMPORTANT: Return ONLY the JSON object. No prose.
"""
        # Tiered models
        models_to_try = [self.model_name, "openai", "mistral-7b"]
        last_err = None

        for model in models_to_try:
            try:
                # If we're using a fallback model, we might need a different client or just change param
                client_to_use = self.client
                if model in ["openai", "mistral-7b"] and "huggingface" in str(self.client.base_url):
                     # Fallback to Pollinations if HF fails
                     from openai import OpenAI
                     client_to_use = OpenAI(base_url="https://text.pollinations.ai/openai", api_key="not-needed")

                response = client_to_use.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=1500,
                    timeout=60
                )
                content = response.choices[0].message.content.strip()
                start = content.find("{")
                end = content.rfind("}")
                if start == -1 or end == -1: continue
                
                return json.loads(content[start:end+1])
            except Exception as e:
                last_err = e
                logger.warning(f"Synthetic generation failed for model {model}: {e}")
                continue
        
        raise last_err or Exception("All generation models failed")
