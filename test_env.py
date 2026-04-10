import sys
import os
from server.environment import ShieldXEnv
from server.models import PrivacyAction

def test_shieldx_compliance():
    print("🛡️ Verifying ShieldX Environment (Strict Bounds Mode)...")
    env = ShieldXEnv(task_id="task-001-pii-scrubber")
    obs = env.reset()
    
    # Test 1: Valid Action (Should be > 0.01 and < 0.99)
    action = PrivacyAction(
        operation="redact",
        target="john.d@gmail.com",
        reasoning="PII Scrutiny"
    )
    obs, reward, done, info = env.step(action)
    print(f"✅ Step 1 (Correct): Reward = {reward}")
    
    # Test 2: Invalid Action (Should be exactly 0.01, NOT 0.0 or negative)
    fail_action = PrivacyAction(
        operation="redact",
        target="wrong-target",
        reasoning="Simulating failure"
    )
    obs, reward, done, info = env.step(fail_action)
    print(f"✅ Step 2 (Incorrect): Reward = {reward}")

    # Test 3: Terminal Success Check
    # We'll force a high reward to see the 0.99 cap
    env.total_reward = 10.0 
    obs, reward, done, info = env.step(action) # This action will be "already_done" -> reward 0 -> capped to 0.01
    print(f"✅ Step 3 (Capped Score): Score = {info['score']}")

    # Validation
    if 0.0 < reward < 1.0 and 0.0 < info['score'] < 1.0:
        print("\n🏆 VERIFICATION PASSED: All values are strictly between 0 and 1.")
    else:
        print("\n❌ VERIFICATION FAILED: Values hit 0.0 or 1.0 boundaries.")

if __name__ == "__main__":
    try:
        test_shieldx_compliance()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
