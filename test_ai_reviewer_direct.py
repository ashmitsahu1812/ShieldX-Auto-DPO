import os
from server.ai_reviewer import analyze_pr

dummy_pr_data = {
    "metadata": {
        "title": "Fix bug",
        "author": "Ashmit",
        "description": "Fixing things",
        "head_branch": "patch-1",
        "base_branch": "main",
        "mergeable_state": "clean",
        "additions": 10,
        "deletions": 5,
        "changed_files": 1
    },
    "files": [
        {
            "filename": "hello.py",
            "status": "modified",
            "additions": 10,
            "deletions": 5,
            "patch": "+print('hello world')"
        }
    ]
}

print("Running analyze_pr test...")
result = analyze_pr(dummy_pr_data)
print("Result:")
print(result)
