import json

INPUT = "channel-announcements.json"
OUTPUT = "clean-announcements.jsonl"

def extract_role(author):
    # pick the highest role by position
    roles = author.get("roles", [])
    if not roles:
        return None
    return max(roles, key=lambda r: r.get("position", 0)).get("name")

with open(INPUT, "r", encoding="utf-8") as f:
    raw = json.load(f)

messages = raw.get("messages", [])

with open(OUTPUT, "w", encoding="utf-8") as out:
    for msg in messages:
        clean = {
            "message_id": msg.get("id"),
            "timestamp": msg.get("timestamp"),
            "channel": raw.get("channel", {}).get("name"),
            "author": msg.get("author", {}).get("nickname")
                    or msg.get("author", {}).get("name"),
            "role": extract_role(msg.get("author", {})),
            "content": msg.get("content", "").strip(),
            "attachments": [
                att.get("fileName")
                for att in msg.get("attachments", [])
            ] if msg.get("attachments") else []
        }
        
        out.write(json.dumps(clean) + "\n")

print(f"✓ Cleaned file written to {OUTPUT}")
