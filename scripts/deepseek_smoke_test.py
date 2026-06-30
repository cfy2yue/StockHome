from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_client import chat_json, extract_json_content, get_api_key, smoke_test_payload


def main() -> None:
    get_api_key()
    print("DEEPSEEK_API_KEY loaded: yes")
    response = chat_json(smoke_test_payload(), max_tokens=256)
    parsed = extract_json_content(response)
    print(json.dumps({"ok": True, "model_response": parsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
