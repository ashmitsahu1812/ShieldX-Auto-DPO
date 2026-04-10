# ShieldX: The Autonomous Data Privacy Officer (DPO)

ShieldX is a real-world Reinforcement Learning environment for automated privacy governance. Agents are tasked with auditing databases, fulfilling Data Subject Access Requests (DSARs), and managing regulatory compliance (GDPR/CCPA/DPA) in high-stakes clinical and corporate environments.

---

## 🛡️ Environment Overview

Privacy engineering is one of the most critical challenges in the modern AI stack. ShieldX provides a benchmark for agents to prove they can:
1. Identify and redact PII accurately.
2. Balance Erasure rights with Legal Retention obligations.
3. Detect Cross-Border compliance violations.

### 📐 Observation Space
The agent receives a `PrivacyObservation` containing:
- **Instruction**: The legal mandate (e.g., "Right to Erasure").
- **Data Buffer**: Raw PII, logs, or JSON records.
- **Policy Context**: The applicable "Laws" for the current region.
- **Region**: Geolocation of the data (EU, US, etc.).

### 🎮 Action Space
- `redact`: Scrub PII from a field.
- `delete`: HARD delete a record.
- `export`: Extract data for a DSAR.
- `retain`: Mark data for legal hold (overrides deletion).
- `notify`: Generate a breach disclosure list.

---

## 🎯 Tasks (5 Baseline Tasks)
| Task ID | Name | Difficulty | Description |
|:---|:---|:---|:---|
| `task-001` | **PII Scrubber** | Easy | Redact Names, SSNs, and IPs from tickets. |
| `task-002` | **DSAR Export** | Medium | Consolidate one user's data; leak zero others. |
| `task-003` | **Selective Erasure** | Medium | Delete Profile, keep Invoices (Tax Law). |
| `task-004` | **Border Audit** | Hard | Audit SCC-less EU->US transfers. |
| `task-005` | **Breach Assessment** | Hard | ID all exfiltrated user IDs from SQLi logs. |

---

## 📈 Reward Design
ShieldX uses a **Trajectorial Reward Function**:
- **Positive Reward (+0.25 to +1.0)**: Correct compliance action.
- **Negative Reward (-0.1 to -0.5)**: Legal violations (e.g., deleting tax records or missed PII).
- **Episode Terminal**: Success (score of 1.0) or exceeding max steps (5).

---

## 🚀 Setup & Usage

### Local Build
```bash
docker build -t shieldx .
docker run -p 7860:7860 shieldx
```

### Validation
```bash
openenv validate
```

### Baseline Inference
```bash
python3 inference.py
```

### Environment Variables
- `HF_TOKEN`: Required for model inference.
- `API_BASE_URL`: Hugging Face Router or OpenAI compatible endpoint.
- `MODEL_NAME`: e.g., `Qwen/Qwen2.5-Coder-32B-Instruct`.

---
*ShieldX is optimized for the OpenEnv Hackathon Challenge.*
