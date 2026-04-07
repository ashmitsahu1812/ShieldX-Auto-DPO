from server.environment import CodeReviewEnv
from server.models import Action
import json

def run_demo():
    print("=== 🛡️ OpenEnv Code Review Simulator: Production Grader Demo ===\n")
    print("Scenario: 'Perfect Agent' identifies a mutable default argument bug.")
    
    # Task: Syntax Review (Mutable List Bug index 1)
    env = CodeReviewEnv(task_type="syntax_review", task_index=1)
    obs = env.state()
    
    print(f"Task: {obs.title}")
    
    print("\n[Step 1] Perfect Agent issues a detailed comment (>= 5 words):")
    action1 = Action(
        action_type="comment",
        file="data.py",
        line=10,
        comment="The use of 'results=[]' is a mutable default argument bug in Python which persists state across calls."
    )
    _, reward1, _, _ = env.step(action1)
    print(f"  > Step Reward: {reward1:+.2f} (Base +0.4, Explanation +0.2, Step -0.05)")
    
    print("\n[Step 2] Perfect Agent then requests changes with justification:")
    action2 = Action(
        action_type="request_changes",
        comment="Requested changes due to detected mutable default argument bug."
    )
    _, reward2, done, info = env.step(action2)
    print(f"  > Step Reward: {reward2:+.2f} (Correct decision +0.5, Step -0.05)")
    
    print(f"\n🏆 Final Episode Score: {info.score:.2f}/1.0")
    print("-" * 60)

    print("Scenario: 'Spam Agent' tries to dump keywords with a short comment.")
    env.reset()
    action_spam = Action(
        action_type="comment",
        file="data.py",
        comment="results=[] bug" # < 5 words
    )
    _, reward_spam, _, _ = env.step(action_spam)
    print(f"  > Step Reward: {reward_spam:+.2f} (Short comment penalty -0.2, Step -0.05)")

    print("\nScenario: 'Lazy Agent' approves a buggy PR without review.")
    env.reset()
    action_lazy = Action(
        action_type="approve",
        comment="Looks good!"
    )
    _, reward_lazy, _, info_lazy = env.step(action_lazy)
    print(f"  > Step Reward: {reward_lazy:+.2f} (Catastrophic decision -0.6, Step -0.05)")
    print(f"  > Missed Bug Penalty Applied (finalize_episode)")
    print(f"🏆 Final Episode Score: {info_lazy.score:.2f}/1.0")
    
    print("\nDemo complete! The production grader successfully prevents exploits and rewards quality review.")

if __name__ == "__main__":
    run_demo()
