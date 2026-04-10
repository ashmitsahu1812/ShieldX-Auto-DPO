import os
from server.environment import ShieldXEnv
from server.models import PrivacyAction

def test_rewards(actions_list):
    env = ShieldXEnv()
    env.reset()
    total_reward = 0.0
    for action_data in actions_list:
        action = PrivacyAction(**action_data)
        _, reward, _, _ = env.step(action)
        total_reward += reward
        print(f"Step Result: {reward:.4f}, Running Total: {total_reward:.4f}")
    
    print(f"Final Cumulative Reward: {total_reward:.4f}")
    assert 0.0 < total_reward < 1.0, f"FAILED: Reward {total_reward} out of range!"
    print("Verification PASSED")

if __name__ == "__main__":
    print("--- Test 1: Perfect Play ---")
    perfect_actions = [{"operation": "redact", "target": "John Doe", "reasoning": "test"}] * 5
    test_rewards(perfect_actions)

    print("\n--- Test 2: Zero Play ---")
    zero_actions = [{"operation": "retain", "target": "nothing", "reasoning": "test"}] * 5
    test_rewards(zero_actions)
