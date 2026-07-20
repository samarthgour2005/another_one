"""
Quick manual smoke test against a running instance of the API.

Usage:
    python scripts/smoke_test.py path/to/some.pdf "What is this document about?"
"""

import sys
import time

import httpx

BASE_URL = "http://localhost:8000/api/v1"


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/smoke_test.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]

    with httpx.Client(timeout=120.0) as client:
        print("Uploading...")
        with open(pdf_path, "rb") as f:
            resp = client.post(f"{BASE_URL}/upload/", files={"files": f})
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        print(f"Session: {session_id}")

        print("Waiting for ingestion to complete...")
        while True:
            docs = client.get(f"{BASE_URL}/sessions/{session_id}/documents").json()["documents"]
            statuses = [d["status"] for d in docs]
            print("  status:", statuses)
            if all(s in ("completed", "failed") for s in statuses):
                break
            time.sleep(2)

        print("Asking question...")
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/",
            json={"session_id": session_id, "message": question},
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    print(line)


if __name__ == "__main__":
    main()
