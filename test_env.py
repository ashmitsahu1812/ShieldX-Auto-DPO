import sys
import os
from server.environment import ShieldXEnv
from server.models import PrivacyAction

def test_shieldx():
    print("🚀 Initializing ShieldX Environment Test...")
    env = ShieldXEnv(task_id="task-001-pii-scrubber")
    obs = env.reset()
    
    print("\n--- Reset Output (Observation) ---")
    print(f"Task: {obs.task_id}")
    print(f"Region: {obs.region}")
    print(f"Instruction: {obs.instruction}")
    print(f"Data Sample: {obs.data_buffer[:100]}...")
    
    # Simulating a compliance action
    action = PrivacyAction(
        operation="redact",
        target="john.d@gmail.com",
        reasoning="Compliance scrubbing of email PII."
    )
    
    print("\n--- Stepping into Environment ---")
    obs, reward, done, info = env.step(action)
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Status: {info.get('message', 'No message')}")
    print(f"New Score: {env.total_score}")
    
    if reward > 0:
        print("\n✅ PROJECT IS WORKING: Reward correctly assigned for valid action.")
    else:
        print("\n❌ PROJECT STATUS: Reward was 0. Check if target exists in data buffer.")

if __name__ == "__main__":
    try:
        test_shieldx()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
