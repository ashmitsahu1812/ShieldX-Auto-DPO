import gradio as gr
from env.environment import CodeReviewEnv
from env.models import Action
from github_fetcher import fetch_full_pr
from ai_reviewer import analyze_pr
import json
import uuid

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
        []
    )

def handle_custom_reset(title, desc, filename, diff, bug_file, bug_line, bug_desc):
    global env
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
        []
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


# ============================================================
# 🌐 LIVE PR REVIEW — Real-World GitHub Integration
# ============================================================

# State for the live review session
live_pr_data = None
live_ai_result = None

def fetch_live_pr(pr_url):
    """Fetches real PR data from GitHub. No dummy data."""
    global live_pr_data, live_ai_result
    live_ai_result = None  # Reset previous analysis

    if not pr_url or "github.com" not in pr_url or "/pull/" not in pr_url:
        return (
            "❌ **Invalid URL.** Please enter a valid GitHub PR link.\n\nExample: `https://github.com/owner/repo/pull/1`",
            "", "", ""
        )
    
    live_pr_data = fetch_full_pr(pr_url)
    
    if "error" in live_pr_data:
        return (
            f"❌ **Error:** {live_pr_data['error']}",
            "", "", ""
        )
    
    meta = live_pr_data["metadata"]
    files = live_pr_data["files"]
    
    # Build PR Info display
    pr_info = f"""## 📋 {meta['title']}
**Author:** `{meta['author']}` • **Branch:** `{meta['head_branch']}` → `{meta['base_branch']}`
**Status:** `{meta['state']}` • **Mergeable:** `{meta['mergeable_state']}`
**Stats:** +{meta['additions']} additions, -{meta['deletions']} deletions across {meta['changed_files']} file(s)

---
**Description:** {meta['description']}
"""

    # Build Diff display
    diff_display = ""
    for f in files:
        badge = "🟢" if f["status"] == "added" else ("🔴" if f["status"] == "removed" else "🟡")
        diff_display += f"### {badge} {f['filename']} ({f['status']})\n"
        diff_display += f"*+{f['additions']} / -{f['deletions']}*\n"
        diff_display += f"```diff\n{f['patch']}\n```\n\n"
    
    # Merge conflict check
    merge_status = ""
    if meta["mergeable_state"] == "clean":
        merge_status = "✅ **No merge conflicts detected.** This PR can be merged cleanly."
    elif meta["mergeable"] is False:
        merge_status = "⚠️ **Merge conflicts detected!** This PR cannot be merged without resolving conflicts."
    else:
        merge_status = f"ℹ️ Merge status: `{meta['mergeable_state']}`"

    return (
        pr_info,
        diff_display,
        merge_status,
        f"✅ Successfully fetched {len(files)} file(s) from GitHub. Click **🤖 Run AI Review** to analyze."
    )


def run_ai_review():
    """Sends the fetched PR diff to Qwen2.5-Coder for analysis."""
    global live_pr_data, live_ai_result
    
    if live_pr_data is None or "error" in live_pr_data:
        return "❌ No PR data loaded. Please fetch a PR first.", ""
    
    live_ai_result = analyze_pr(live_pr_data)
    
    # Format AI comments
    comments_display = "## 🤖 AI Review Results\n\n"
    
    for i, comment in enumerate(live_ai_result.get("comments", []), 1):
        severity = comment.get("severity", "info")
        icon = "🔴" if severity == "error" else ("🟡" if severity == "warning" else "🔵")
        comments_display += f"### {icon} Finding #{i} — `{comment['file']}`\n"
        comments_display += f"**Severity:** {severity.upper()}\n\n"
        comments_display += f"{comment['comment']}\n\n---\n\n"
    
    # Verdict
    verdict = live_ai_result.get("overall_verdict", "unknown")
    verdict_reason = live_ai_result.get("verdict_reason", "")
    verdict_icon = "✅" if verdict == "approve" else "❌"
    
    verdict_display = f"""## {verdict_icon} AI Verdict: **{verdict.upper()}**

{verdict_reason}

> **Note:** This is the AI's recommendation. The final decision is yours — use the buttons below to Approve or Reject.
"""
    
    return comments_display, verdict_display


def user_approve():
    """User decides to approve the PR."""
    if live_pr_data is None:
        return "No PR loaded."
    meta = live_pr_data["metadata"]
    return f"""## ✅ PR APPROVED by You

**PR:** {meta['title']}
**Author:** {meta['author']}
**Your Decision:** APPROVE

You can now safely merge this PR on GitHub:
👉 [{meta['html_url']}]({meta['html_url']})
"""


def user_reject():
    """User decides to reject the PR."""
    if live_pr_data is None:
        return "No PR loaded."
    meta = live_pr_data["metadata"]
    
    # Include AI findings in rejection reason
    findings = ""
    if live_ai_result and "comments" in live_ai_result:
        for c in live_ai_result["comments"]:
            if c["severity"] in ["error", "warning"]:
                findings += f"- **{c['file']}**: {c['comment']}\n"
    
    return f"""## ❌ PR REJECTED by You

**PR:** {meta['title']}
**Author:** {meta['author']}
**Your Decision:** REQUEST CHANGES

### Issues to Address:
{findings if findings else "- No specific AI findings. You rejected based on your own judgment."}

Share this feedback with the author on GitHub:
👉 [{meta['html_url']}]({meta['html_url']})
"""


# ============================================================
# 🎨 GRADIO UI LAYOUT
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="ScalarX Meta — AI Code Review") as demo:
    
    with gr.Tabs() as main_tabs:
        
        # ====== TAB 1: Live PR Review (NEW — Real World) ======
        with gr.TabItem("🌐 Live PR Review"):
            gr.Markdown("# 🌐 Live GitHub PR Review")
            gr.Markdown("Paste a real GitHub PR URL → Fetch the code → Let AI analyze it → You make the final call.")
            
            with gr.Row():
                pr_url_input = gr.Textbox(
                    label="GitHub PR URL",
                    placeholder="https://github.com/owner/repo/pull/1",
                    scale=4
                )
                fetch_btn = gr.Button("📥 Fetch PR", variant="primary", scale=1)
            
            with gr.Row():
                with gr.Column(scale=1):
                    live_pr_info = gr.Markdown("Enter a GitHub PR URL above and click **Fetch PR**.")
                    live_merge_status = gr.Markdown("")
                    live_status = gr.Markdown("")
                    
                    gr.Markdown("---")
                    analyze_btn = gr.Button("🤖 Run AI Review", variant="secondary", size="lg")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🧑‍💻 Your Decision")
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve PR", variant="primary")
                        reject_btn = gr.Button("❌ Reject PR", variant="stop")
                    user_decision_output = gr.Markdown("")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("📄 Code Changes"):
                            live_diff_view = gr.Markdown("Diff will appear here after fetching.")
                        with gr.TabItem("🤖 AI Analysis"):
                            ai_comments_output = gr.Markdown("Click **Run AI Review** after fetching a PR.")
                            ai_verdict_output = gr.Markdown("")
            
            # Event handlers for Live PR
            fetch_btn.click(
                fetch_live_pr,
                inputs=[pr_url_input],
                outputs=[live_pr_info, live_diff_view, live_merge_status, live_status]
            )
            analyze_btn.click(
                run_ai_review,
                inputs=[],
                outputs=[ai_comments_output, ai_verdict_output]
            )
            approve_btn.click(user_approve, inputs=[], outputs=[user_decision_output])
            reject_btn.click(user_reject, inputs=[], outputs=[user_decision_output])
        
        # ====== TAB 2: OpenEnv Simulator (Original) ======
        with gr.TabItem("🛡️ OpenEnv Simulator"):
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
                            cust_diff = gr.Code(label="Unified Diff (.patch style)", language="markdown", lines=10)
                            
                            gr.Markdown("---")
                            gr.Markdown("### 🎯 Grader Metadata (How to score)")
                            with gr.Row():
                                bug_file = gr.Textbox(label="Bug File Name", placeholder="auth.py")
                                bug_line = gr.Number(label="Bug Line Number", value=0, precision=0)
                            bug_desc = gr.Textbox(label="Bug Description (Expected Explanation)")
                            
                            load_custom_btn = gr.Button("🚀 Load Custom Challenge", variant="primary")

            # Event handlers for OpenEnv Simulator
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
