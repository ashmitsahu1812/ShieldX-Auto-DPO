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
        # print(f"Step Result: {reward:.4f}, Running Total: {total_reward:.4f}")
    
    print(f"Final Cumulative Reward: {total_reward:.4f}")
    assert 0.0 < total_reward < 1.0, f"FAILED: Reward {total_reward} out of range!"
    return total_reward

if __name__ == "__main__":
    print("--- Test 1: Diverse Perfect Play ---")
    # Ground truth for task 1: ['John Doe', 'john.d@gmail.com', '999-00-1111', '192.168.1.1', 'Jane Smith', '10.0.0.45']
    diverse_actions = [
        {"operation": "redact", "target": "John Doe", "reasoning": "test"},
        {"operation": "redact", "target": "john.d@gmail.com", "reasoning": "test"},
        {"operation": "redact", "target": "999-00-1111", "reasoning": "test"},
        {"operation": "redact", "target": "Jane Smith", "reasoning": "test"},
        {"operation": "redact", "target": "10.0.0.45", "reasoning": "test"}
    ]
    test_rewards(diverse_actions)
