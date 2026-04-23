#!/usr/bin/env python3
"""
Session Storage for DriftDetector v2 (Internal Use Only)

Stores drift reports organized by session (date-based sessions).
This is NOT part of the OpenSource release - users manage their own storage.

For OpenSource: Users decide if/how to persist data.
For Internal: Use this for session-based dashboard tracking.
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional
import json


class SessionStorage:
    """Manage drift sessions and reports"""

    def __init__(self, db_path: str = "drift_sessions.db"):
        """Initialize session storage"""
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if not exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Sessions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_reports INTEGER DEFAULT 0,
                avg_drift REAL,
                max_drift REAL,
                min_drift REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Reports table (organized by session)
        c.execute('''
            CREATE TABLE IF NOT EXISTS session_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_number INTEGER,
                timestamp TEXT NOT NULL,
                combined_drift_score REAL,
                ghost_loss REAL,
                behavior_shift REAL,
                agreement_score REAL,
                stagnation_score REAL,
                is_drifting BOOLEAN,
                metadata JSON,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')

        # Index for faster queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON session_reports(session_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON session_reports(timestamp)')

        conn.commit()
        conn.close()

    def create_session(self, session_date: Optional[date] = None) -> str:
        """Create new session"""
        session_date = session_date or date.today()
        session_id = f"session_{session_date.isoformat()}"

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT OR IGNORE INTO sessions (session_id, session_date, start_time)
            VALUES (?, ?, ?)
        ''', (session_id, session_date.isoformat(), datetime.now().isoformat()))

        conn.commit()
        conn.close()

        return session_id

    def add_report(self, session_id: str, report: Dict):
        """Add drift report to session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT INTO session_reports (
                session_id, step_number, timestamp, combined_drift_score,
                ghost_loss, behavior_shift, agreement_score, stagnation_score,
                is_drifting, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            report.get('step_number'),
            report.get('timestamp'),
            report.get('combined_drift_score'),
            report.get('ghost_loss'),
            report.get('behavior_shift'),
            report.get('agreement_score'),
            report.get('stagnation_score'),
            report.get('is_drifting'),
            json.dumps(report.get('metadata', {}))
        ))

        conn.commit()
        conn.close()

    def get_session_reports(self, session_id: str) -> List[Dict]:
        """Get all reports for a session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT * FROM session_reports WHERE session_id = ?
            ORDER BY timestamp DESC
        ''', (session_id,))

        columns = [desc[0] for desc in c.description]
        reports = [dict(zip(columns, row)) for row in c.fetchall()]

        conn.close()
        return reports

    def get_sessions(self) -> List[Dict]:
        """Get all sessions with stats"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('SELECT * FROM sessions ORDER BY session_date DESC')
        columns = [desc[0] for desc in c.description]
        sessions = [dict(zip(columns, row)) for row in c.fetchall()]

        conn.close()
        return sessions

    def update_session_stats(self, session_id: str):
        """Update session statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT COUNT(*), AVG(combined_drift_score), MAX(combined_drift_score), MIN(combined_drift_score)
            FROM session_reports WHERE session_id = ?
        ''', (session_id,))

        total, avg, max_val, min_val = c.fetchone()

        c.execute('''
            UPDATE sessions SET total_reports = ?, avg_drift = ?, max_drift = ?, min_drift = ?
            WHERE session_id = ?
        ''', (total, avg or 0.0, max_val or 0.0, min_val or 0.0, session_id))

        conn.commit()
        conn.close()

    def close_session(self, session_id: str):
        """Close session (set end_time)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            UPDATE sessions SET end_time = ? WHERE session_id = ?
        ''', (datetime.now().isoformat(), session_id))

        conn.commit()
        conn.close()


# Example Usage (for internal testing)
if __name__ == "__main__":
    storage = SessionStorage()

    # Create session
    session_id = storage.create_session()
    print(f"Created session: {session_id}")

    # Add sample reports
    for i in range(5):
        storage.add_report(session_id, {
            'step_number': i + 1,
            'timestamp': datetime.now().isoformat(),
            'combined_drift_score': 0.3 + (i * 0.1),
            'ghost_loss': 0.2,
            'behavior_shift': 0.1,
            'agreement_score': 0.8,
            'stagnation_score': 0.0,
            'is_drifting': False
        })

    # Update stats
    storage.update_session_stats(session_id)

    # Get reports
    reports = storage.get_session_reports(session_id)
    print(f"Reports in session: {len(reports)}")

    # Get all sessions
    sessions = storage.get_sessions()
    print(f"Total sessions: {len(sessions)}")
