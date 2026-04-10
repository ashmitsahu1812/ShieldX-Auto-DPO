import json
from typing import List, Dict, Any

TASKS = [
    {
        "id": "task-001-pii-scrubber",
        "name": "PII Discovery and Redaction",
        "difficulty": "easy",
        "instruction": "Scan the provided support ticket log. Redact any PII found (Name, Email, SSN, IP Address).",
        "data": "Ticket #102: User 'John Doe' (email: john.d@gmail.com, SSN: 999-00-1111) is reporting a login error from IP 192.168.1.1. Also, customer 'Jane Smith' at 10.0.0.45 requested a password reset.",
        "region": "Asia-South-1 (India)",
        "policy": "DPDP Act Compliance: All PII must be redacted in non-production logs for data fiduciary safety.",
        "ground_truth": ["John Doe", "john.d@gmail.com", "999-00-1111", "192.168.1.1", "Jane Smith", "10.0.0.45"]
    },
    {
        "id": "task-002-dsar-export",
        "name": "Subject Access Request (DSAR)",
        "difficulty": "medium",
        "instruction": "A customer (ID: USER_778) has requested all their data. Filter the database logs and export only their specific entries.",
        "data": json.dumps([
            {"user_id": "USER_778", "activity": "login", "timestamp": "2026-04-10T10:00:00", "ip": "1.1.1.1"},
            {"user_id": "USER_123", "activity": "purchase", "timestamp": "2026-04-10T10:05:00", "ip": "2.2.2.2"},
            {"user_id": "USER_778", "activity": "update_profile", "timestamp": "2026-04-10T10:10:00", "ip": "1.1.1.1"},
            {"user_id": "USER_778", "activity": "logout", "timestamp": "2026-04-10T10:15:00", "ip": "1.1.1.1"}
        ]),
        "region": "Asia-South-1 (India)",
        "policy": "Right of Access: Data principals must receive a copy of their personal data only.",
        "ground_truth": ["USER_778"]
    },
    {
        "id": "task-003-selective-erasure",
        "name": "The Right to be Forgotten (Selective)",
        "difficulty": "medium",
        "instruction": "Fulfill the erasure request for 'Alice'. Delete her account profile but RETAIN her billing record for auditing purposes.",
        "data": json.dumps({
            "profile": {"name": "Alice Smith", "age": 28, "pref": "Dark Mode", "joined": "2024-01-01"},
            "billing": {"invoice_id": "INV-2026-001", "amount": 99.99, "status": "paid", "due_date": "2026-05-01"}
        }),
        "region": "Asia-South-1 (India)",
        "policy": "Retention Exception: Retention is permitted for financial/tax compliance under Indian Law.",
        "ground_truth_delete": ["profile"],
        "ground_truth_retain": ["billing"]
    },
    {
        "id": "task-004-cross-border-audit",
        "name": "Cross-Border Transfer Compliance",
        "difficulty": "hard",
        "instruction": "Detect if Indian user data is being transferred abroad without proper SCC flag. If found, RETAIN and mark for Audit.",
        "data": json.dumps([
            {"src": "IN", "dst": "US", "id": "X-001", "scc": True, "size": "1.2MB"},
            {"src": "IN", "dst": "EU", "id": "X-002", "scc": False, "size": "0.5MB"},
            {"src": "IN", "dst": "SG", "id": "X-003", "scc": False, "size": "4.8MB"}
        ]),
        "region": "Asia-South-1 (India)",
        "policy": "Data Sovereignty: Explicit SCCs are mandatory for international data transfers.",
        "ground_truth": ["X-002", "X-003"]
    },
    {
        "id": "task-005-breach-reporting",
        "name": "Automated Breach Disclosure",
        "difficulty": "hard",
        "instruction": "Identify all distinct users affected by the SQL injection attack in this log snippet.",
        "data": "LOG: 2026-04-10 12:00:05 - SQL_INJECTION - SELECT * FROM users WHERE id IN (101, 102, 105, 107, 110, 112) - EXFILTRATED",
        "region": "Asia-South-1 (India)",
        "policy": "CERT-In Notification: Identify IDs requiring disclosure within the 6-hour reporting window.",
        "ground_truth": ["101", "102", "105", "107", "110", "112"]
    }
]

def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    return TASKS[0]
