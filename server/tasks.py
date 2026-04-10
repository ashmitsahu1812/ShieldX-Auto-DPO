import json
from typing import List, Dict, Any

TASKS = [
    {
        "id": "task-001-pii-scrubber",
        "name": "PII Discovery and Redaction",
        "difficulty": "easy",
        "instruction": "Scan the provided support ticket log. Redact any PII found (Name, Email, SSN, IP Address).",
        "data": "Ticket #102: User 'John Doe' (email: john.d@gmail.com, SSN: 999-00-1111) is reporting a login error from IP 192.168.1.1.",
        "region": "US-East-1",
        "policy": "GDPR Compliance: All PII must be redacted in non-production logs.",
        "ground_truth": ["John Doe", "john.d@gmail.com", "999-00-1111", "192.168.1.1"]
    },
    {
        "id": "task-002-dsar-export",
        "name": "Subject Access Request (DSAR)",
        "difficulty": "medium",
        "instruction": "A customer (ID: USER_778) has requested all their data. Filter the database logs and export only their specific entries.",
        "data": json.dumps([
            {"user_id": "USER_778", "activity": "login", "timestamp": "2026-04-10T10:00:00"},
            {"user_id": "USER_123", "activity": "purchase", "timestamp": "2026-04-10T10:05:00"},
            {"user_id": "USER_778", "activity": "logout", "timestamp": "2026-04-10T10:10:00"}
        ]),
        "region": "EU-West-1",
        "policy": "Right of Access: Users must receive a copy of their personal data only.",
        "ground_truth": ["USER_778"]
    },
    {
        "id": "task-003-selective-erasure",
        "name": "The Right to be Forgotten (Selective)",
        "difficulty": "medium",
        "instruction": "Fulfill the erasure request for 'Alice'. Delete her account profile but RETAIN her billing record for tax auditing purposes.",
        "data": json.dumps({
            "profile": {"name": "Alice Smith", "age": 28, "pref": "Dark Mode"},
            "billing": {"invoice_id": "INV-2026-001", "amount": 99.99, "status": "paid"}
        }),
        "region": "US-West-2",
        "policy": "Erasure Exception: Retention is permitted for legal/tax compliance.",
        "ground_truth_delete": ["profile"],
        "ground_truth_retain": ["billing"]
    },
    {
        "id": "task-004-cross-border-audit",
        "name": "Cross-Border Transfer Compliance",
        "difficulty": "hard",
        "instruction": "Detect if EU user data is being transferred to US without the 'SCC' (Standard Contractual Clauses) flag. If found, RETAIN and mark for Audit.",
        "data": json.dumps([
            {"src": "EU", "dst": "US", "id": "X-001", "scc": True},
            {"src": "EU", "dst": "US", "id": "X-002", "scc": False}
        ]),
        "region": "Global",
        "policy": "Data Residency: SCCs are mandatory for EU->US transfers.",
        "ground_truth": ["X-002"]
    },
    {
        "id": "task-005-breach-reporting",
        "name": "Automated Breach Disclosure",
        "difficulty": "hard",
        "instruction": "Identify all distinct users affected by the SQL injection attack in this log snippet.",
        "data": "LOG: 2026-04-10 12:00:05 - SQL_INJECTION - SELECT * FROM users WHERE id IN (101, 102, 105, 107) - EXFILTRATED",
        "region": "US-East-1",
        "policy": "Data Breach Notification: Identify exactly which IDs require disclosure.",
        "ground_truth": ["101", "102", "105", "107"]
    }
]

def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    return TASKS[0]
