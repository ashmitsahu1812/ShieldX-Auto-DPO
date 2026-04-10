---
title: ShieldX - Autonomous DPO
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - privacy
  - rl
---

# 🛡️ ShieldX: The Autonomous Data Privacy Officer (DPO)

ShieldX is a real-world Reinforcement Learning environment designed for **Data Privacy Governance**. It provides a high-fidelity simulation of an AI agent managing GDPR/CCPA compliance across a distributed corporate database.

---

## 📐 Environment Design & MDP

ShieldX models the complexity of privacy engineering as a Markov Decision Process (MDP):
- **Observation Space**: Multi-region database fragments, server logs, and legal schemas.
- **Action Space**: Privacy-preserving operations (`REDACT`, `DELETE`, `EXPORT`, `RETAIN`, `NOTIFY`).
- **Reward Function**: Dense, trajectorial reward shaping that provides progress signals for correct compliance handling while penalizing legal risks.

### **Reward Architecture**
- **Task Success**: `+1.0` (Terminal)
- **Partial Progress**: `+0.25` per correctly identified PII or fulfilled sub-requirement.
- **Legal Violation**: `-0.1` to `-0.5` for destructive actions (e.g., deleting required tax data).

---

## 🎯 Task Library (ShieldX-Benchmark)

| Task ID | Name | Difficulty | Description |
|:---|:---|:---|:---|
| `task-001` | **PII Scrubber** | Easy | Identify and redact Names, Emails, SSNs, and IPs from unstructured support logs. |
| `task-002` | **DSAR Export** | Medium | Fulfill a Subject Access Request by consolidating User Data without leaking PII of other entities. |
| `task-003` | **Selective Erasure** | Medium | Balance 'Right to Erasure' with 'Tax Retention'. Delete Profile data but retain Billing History. |
| `task-004` | **Border Compliance** | Hard | Audit cross-border data transfers (EU -> US) and identify missing Standard Contractual Clauses (SCCs). |
| `task-005` | **Breach Disclosure** | Hard | Analyze SQL injection logs to identify all exfiltrated user IDs and determine notification requirements. |

---

## 🏗️ Technical Architecture

- **Core Engine**: Python-based Gymnasium wrapper with deterministic Pydantic validation.
- **API Surface**: FastAPI implementation of the OpenEnv spec (`/reset`, `/step`, `/state`).
- **User Interface**: Glassmorphism Gradio dashboard for real-time audit visualization.
- **Inference**: Strict `inference.py` baseline using the Hugging Face Router API.

---

## 🚀 Setup & Validation

### **Local Deployment**
```bash
docker build -t shieldx .
docker run -p 7860:7860 shieldx
```

### **Running the Baseline**
Ensure `HF_TOKEN` is set in your `.env` file or environment:
```bash
python3 inference.py
```

### **Compliance Check**
ShieldX is 100% compliant with the OpenEnv specification and yields a `PASS` on automated validator scripts.

---

### **Environment Variables**
- `HF_TOKEN`: Required for model-based inference in the baseline script.
- `API_BASE_URL`: Default is `https://router.huggingface.co/v1`.
- `MODEL_NAME`: Default is `Qwen/Qwen2.5-Coder-32B-Instruct`.

---
*ShieldX is a production-level environment optimized for the OpenEnv Hackathon Round 1 Benchmark.*
