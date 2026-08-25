"""
Logger for recording all LLM calls to llm_calls.jsonl.
Strictly conforms to required schema:
{
  "stage": "string",
  "query_id": "string | null",
  "timestamp": "ISO-8601 timestamp",
  "provider": "string",
  "model": "string",
  "prompt_hash": "string",
  "input_artifacts": ["path"],
  "output_artifact": "path"
}
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, List, Optional


class LLMLogger:
    def __init__(self, log_filepath: str = "llm_calls.jsonl", clear_existing: bool = True):
        self.log_filepath = log_filepath
        if clear_existing and os.path.exists(log_filepath):
            os.remove(log_filepath)

    def log_call(
        self,
        stage: str,
        query_id: Optional[str],
        provider: str,
        model: str,
        prompt_content: Any,
        input_artifacts: List[str],
        output_artifact: str,
    ) -> Dict[str, Any]:
        """Logs a single LLM invocation to llm_calls.jsonl."""
        # Convert prompt content to string for deterministic hashing
        if isinstance(prompt_content, (dict, list)):
            prompt_str = json.dumps(prompt_content, sort_keys=True)
        else:
            prompt_str = str(prompt_content)

        prompt_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "stage": stage,
            "query_id": query_id,
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
            "input_artifacts": input_artifacts,
            "output_artifact": output_artifact,
        }

        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record
