import os
import json
from datetime import datetime
from pathlib import Path

def _get_log_path() -> Path:
    # Save in logs/progress_history.json
    base_dir = Path(__file__).resolve().parent.parent.parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir / "progress_history.json"

def _load_logs() -> dict:
    log_file = _get_log_path()
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[*] Error loading progress logs: {e}")
            
    return {"session_id": "student_svecw_99", "history": []}

def _save_logs(data: dict):
    log_file = _get_log_path()
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[*] Error saving progress logs: {e}")

def append_history(profile: dict, scores: dict):
    data = _load_logs()
    
    # Clean profile to make it serializable
    clean_profile = {
        "cgpa": profile.get("cgpa") if profile.get("cgpa") is not None else None,
        "skills": list(profile.get("skills", [])),
        "weaknesses": list(profile.get("weaknesses", [])),
        "interests": list(profile.get("interests", [])),
        "backlogs": int(profile.get("backlogs", 0)),
        "projects_count": int(profile.get("projects_count", 0))
    }
    
    # Clean scores
    clean_scores = {k: float(v) for k, v in scores.items()}
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "profile": clean_profile,
        "scores": clean_scores
    }
    
    data["history"].append(entry)
    _save_logs(data)

def get_history() -> list:
    data = _load_logs()
    return data.get("history", [])

def get_latest_profile() -> dict:
    history = get_history()
    if history:
        return history[-1].get("profile")
    return None

def clear_history():
    data = {"session_id": "student_svecw_99", "history": []}
    _save_logs(data)
