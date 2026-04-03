import gradio as gr
from env.environment import CodeReviewEnv
from env.models import Action
import json

# Global environment instance for the UI session
env = None

def reset_env(task_type, task_index):
    global env
    env = CodeReviewEnv(task_type=task_type, task_index=int(task_index))
    obs = env.state()
    
    # Format files changed for display
    files_display = ""
    for f in obs.files_changed:
        files_display += f"### 📄 {f.filename}\n```diff\n{f.diff}\n```\n\n"
    
    return (
        f"## {obs.title}\n{obs.description}",
        files_display,
        "Ready for review.",
        0.0,
        False,
        [],
        gr.update(selected=0) # Switch to Overview tab on reset
    )

def handle_custom_reset(title, desc, filename, diff, bug_file, bug_line, bug_desc):
    global env
    # Construct a custom task dictionary
    custom_data = {
        "pr_id": f"custom-{uuid.uuid4().hex[:6]}",
        "title": title or "Custom PR Review",
        "description": desc or "No description provided.",
        "files_changed": [
            {"filename": filename or "main.py", "diff": diff or ""}
        ],
        "expected_bugs": [
            {
                "file": bug_file or filename or "main.py",
                "line": int(bug_line) if bug_line else 0,
                "type": "custom_bug",
                "description": bug_desc or "User defined bug."
            }
        ],
        "expected_action": "request_changes"
    }
    
    env = CodeReviewEnv(task_type="custom", custom_data=custom_data)
    obs = env.state()
    
    files_display = f"### 📄 {filename or 'main.py'}\n```diff\n{diff}\n```\n\n"
    
    return (
        f"## {obs.title}\n{obs.description}",
        files_display,
        "Custom challenge loaded. Start review!",
        0.0,
        False,
        [],
        gr.update(selected=0) # Switch to Overview tab on reset
    )

def handle_action(action_type, comment, file_name, line_num):
    global env
    if env is None:
        return "Please reset the environment first.", "", 0.0, True, []
    
    # Create action object
    if action_type == "comment":
        action = Action(
            action_type="comment",
            comment=comment,
            file=file_name if file_name else None,
            line=int(line_num) if line_num else None
        )
    else:
        action = Action(
            action_type=action_type,
            comment=comment
        )
    
    obs, reward, done, info = env.step(action)
    
    # Format history for the UI table
    history = []
    for h in obs.comments_history:
        history.append(["comment", h, "-", 0])
    
    status_msg = f"Last Action: {action_type.upper()}\nFeedback: {obs.last_action_feedback}"
    if done:
        status_msg += "\n\n🏁 Session Finished!"
    
    return (
        status_msg,
        f"{info.score:.2f}",
        done,
        history
    )

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="OpenEnv Code Review Dashboard") as demo:
    gr.Markdown("# 🛡️ OpenEnv Code Review Simulator")
    gr.Markdown("Test your AI agent's (or your own) code review skills against deterministic adversarial PRs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            task_type = gr.Dropdown(
                label="Task Type", 
                choices=["syntax_review", "bug_detection", "full_review", "adversarial_review"],
                value="syntax_review"
            )
            task_index = gr.Number(label="Task Index", value=0, precision=0)
            reset_btn = gr.Button("🚀 Initialize / Reset", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### 🕹️ Action Space")
            action_type = gr.Radio(
                label="Action", 
                choices=["comment", "approve", "request_changes"], 
                value="comment"
            )
            comment_input = gr.Textbox(label="Comment Text", placeholder="e.g., Found a bug in line 10...")
            
            with gr.Row():
                file_input = gr.Textbox(label="File (Optional)", placeholder="main.py")
                line_input = gr.Number(label="Line (Optional)", value=0, precision=0)
            
            submit_btn = gr.Button("📤 Submit Action", variant="secondary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📋 PR Overview"):
                    pr_info = gr.Markdown("Click Initialize to load a PR.")
                    diff_view = gr.Markdown("")
                
                with gr.TabItem("📊 Execution Logs"):
                    status_output = gr.Textbox(label="Environment Status", interactive=False, lines=5)
                    score_output = gr.Label(label="Current Score / 1.0")
                    history_table = gr.DataFrame(
                        headers=["Action", "Comment", "File", "Line"],
                        label="Action History",
                        datatype=["str", "str", "str", "number"]
                    )
                    is_done = gr.Checkbox(label="Done?", interactive=False)
                
                with gr.TabItem("🛠️ Custom PR Creator"):
                    gr.Markdown("### 🔨 Create Your Own Evaluation Task")
                    with gr.Row():
                        cust_title = gr.Textbox(label="PR Title", placeholder="e.g., Fix security vulnerability")
                        cust_filename = gr.Textbox(label="File Name", placeholder="auth.py")
                    cust_desc = gr.Textbox(label="PR Description", lines=2)
                    cust_diff = gr.Code(label="Unified Diff (.patch style)", language="diff", lines=10)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🎯 Grader Metadata (How to score)")
                    with gr.Row():
                        bug_file = gr.Textbox(label="Bug File Name", placeholder="auth.py")
                        bug_line = gr.Number(label="Bug Line Number", value=0, precision=0)
                    bug_desc = gr.Textbox(label="Bug Description (Expected Explanation)")
                    
                    load_custom_btn = gr.Button("🚀 Load Custom Challenge", variant="primary")

    # --- Event Handlers ---
    reset_btn.click(
        reset_env, 
        inputs=[task_type, task_index], 
        outputs=[pr_info, diff_view, status_output, score_output, is_done, history_table]
    )
    
    submit_btn.click(
        handle_action,
        inputs=[action_type, comment_input, file_input, line_input],
        outputs=[status_output, score_output, is_done, history_table]
    )
    
    load_custom_btn.click(
        handle_custom_reset,
        inputs=[cust_title, cust_desc, cust_filename, cust_diff, bug_file, bug_line, bug_desc],
        outputs=[pr_info, diff_view, status_output, score_output, is_done, history_table]
    )

if __name__ == "__main__":
    demo.launch()
