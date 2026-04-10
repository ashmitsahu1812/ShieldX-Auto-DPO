import json
import os
import sys

# Add parent directory to path so we can import server modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ai_reviewer import analyze_pr

def run_synthetic_tests():
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synthetic_dataset.json")
    
    if not os.path.exists(dataset_path):
        print("synthetic_dataset.json not found.")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} synthetic PR cases.\n")
    
    successes = 0
    
    for case in dataset:
        print(f"--- Running Test: {case['task_id']} ---")
        pr_data = {
            "metadata": {
                "title": case["title"],
                "author": "synthetic_user",
                "description": case["description"],
                "head_branch": "fix-branch",
                "base_branch": "main",
                "mergeable_state": "clean",
                "additions": sum(f.get("additions", 0) for f in case["files_changed"]),
                "deletions": sum(f.get("deletions", 0) for f in case["files_changed"]),
                "changed_files": len(case["files_changed"])
            },
            "files": case["files_changed"]
        }
        
        result = analyze_pr(pr_data)
        
        # Check if the AI verdict matched our expectations
        comments_text = " ".join([c["comment"] for c in result.get("comments", [])]).lower()
        
        passed = False
        for truth in case.get("ground_truth_bugs", []):
            if truth["keyword"].lower() in comments_text:
                passed = True
                break
                
        if passed:
            print(f"✅ PASSED: AI caught the '{truth['keyword']}' bug.")
            successes += 1
        else:
            print(f"❌ FAILED: AI missed the expected bug.")
            print(f"AI Comments: {json.dumps(result.get('comments', []), indent=2)}")
            
        print("\n")
        
    print(f"Total Score: {successes}/{len(dataset)} ({successes/len(dataset)*100:.1f}%)")

if __name__ == "__main__":
    run_synthetic_tests()
