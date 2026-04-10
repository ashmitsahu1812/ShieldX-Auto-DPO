import gradio as gr
import json
import os
from .environment import ShieldXEnv
from .models import PrivacyAction

CUSTOM_CSS = """
.glass-panel { background: rgba(17, 24, 39, 0.7) !important; backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 12px !important; }
.primary-btn { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; border: none !important; color: white !important; font-weight: bold !important; }
.secondary-btn { background: rgba(55, 65, 81, 0.8) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; color: white !important; }
"""

def create_shieldx_demo():
    env = ShieldXEnv()

    def handle_reset(task_id):
        nonlocal env
        env = ShieldXEnv(task_id=task_id)
        obs = env.reset()
        return (
            f"### 🎯 Task: {env.task['name']}\n\n**Difficulty:** {env.task['difficulty'].upper()}\n\n{obs.instruction}",
            obs.data_buffer,
            obs.policy_context,
            f"Region: {obs.region} | Step: 0/{obs.max_steps}",
            ""
        )

    def handle_step(op, target, basis, reason):
        nonlocal env
        action = PrivacyAction(operation=op, target=target, legal_basis=basis, reasoning=reason)
        obs, reward, done, info = env.step(action)
        status = f"### 🔄 Action Result: {info['explanation']}\n\n**Reward:** {reward:+.2f} | **Total Score:** {info['score']:.2f}"
        if done: status += "\n\n🏁 **Episode Finished.**"
        return (obs.data_buffer, f"Region: {obs.region} | Step: {obs.step_count}/{obs.max_steps}", status, json.dumps(env.history, indent=2))

    with gr.Blocks(theme=gr.themes.Glass(), css=CUSTOM_CSS, title="ShieldX — Autonomous DPO") as demo:
        gr.Markdown("# 🛡️ ShieldX: Autonomous Data Privacy Officer")
        with gr.Row():
            with gr.Column(scale=1):
                task_selector = gr.Dropdown(label="Current Compliance Task", choices=["task-001-pii-scrubber", "task-002-dsar-export", "task-003-selective-erasure", "task-004-cross-border-audit", "task-005-breach-reporting"], value="task-001-pii-scrubber")
                reset_btn = gr.Button("🔄 Initialize Audit", variant="primary", elem_classes=["primary-btn"])
                task_info = gr.Markdown("Select a task and click **Initialize Audit**.")
                region_status = gr.Markdown("Region: - | Step: 0/0")
            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                with gr.Tabs():
                    with gr.TabItem("📊 Data Buffer"): data_view = gr.Textbox(label="Live Data", lines=10)
                    with gr.TabItem("⚖️ Policy Context"): policy_view = gr.Textbox(label="Laws & Rules", lines=10)
                    with gr.TabItem("📜 Audit History"): history_view = gr.Code(label="Action Trace", language="json")
        with gr.Row(elem_classes=["glass-panel"]):
            with gr.Column():
                with gr.Row():
                    op_input = gr.Radio(choices=["redact", "delete", "export", "retain", "notify"], label="Operation", value="redact")
                    target_input = gr.Textbox(label="Target (Field/ID)", placeholder="e.g. John Doe", scale=2)
                with gr.Row():
                    basis_input = gr.Textbox(label="Legal Basis", placeholder="GDPR Art. 6(1)(f)", scale=1)
                    reason_input = gr.Textbox(label="Reasoning", placeholder="PII exposure prevention", scale=2)
                step_btn = gr.Button("⚙️ Execute Action", variant="secondary", elem_classes=["secondary-btn"])
                audit_status = gr.Markdown("")
        reset_btn.click(handle_reset, inputs=[task_selector], outputs=[task_info, data_view, policy_view, region_status, audit_status])
        step_btn.click(handle_step, inputs=[op_input, target_input, basis_input, reason_input], outputs=[data_view, region_status, audit_status, history_view])
    return demo
