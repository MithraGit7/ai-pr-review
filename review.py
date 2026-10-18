"""AI PR reviewer: reads a pull request diff, asks a free AI model to review it,
and posts the result as a review on the pull request. Uses only the Python standard library."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

LLM_URL = os.getenv("LLM_URL", "https://models.github.ai/inference/chat/completions")
MODEL = os.getenv("REVIEW_MODEL", "openai/gpt-4o-mini")
TOKEN = os.getenv("GITHUB_TOKEN", "")
REPO = os.getenv("GITHUB_REPOSITORY", "")
PR_NUMBER = os.getenv("PR_NUMBER", "")
MAX_CHARS = 12000

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
SKIP_SUFFIX = (".lock", ".min.js", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".pdf")
SKIP_NAMES = ("package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock")
ICONS = {"high": "🔴", "medium": "🟡", "low": "🔵"}

PROMPT = """You are a senior engineer reviewing a pull request.
Find real problems: bugs, security issues, bad error handling, missing tests, unclear code.
Skip nitpicks and praise. Prefer a few high-value comments over many weak ones.

The diff shows each new-file line as "<line number>: <code>". Only comment on lines
that start with "+" (added lines) and use that line number.

Reply with ONLY valid JSON, no markdown fences:
{"summary": "1-2 sentence summary of this change",
 "comments": [{"path": "file path", "line": 12, "severity": "high|medium|low", "comment": "what is wrong and how to fix it"}]}
If you find nothing worth flagging, return an empty comments list."""


def send(method, url, payload=None, accept="application/vnd.github+json"):
    """Small HTTP helper. Returns (status_code, response_text)."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": "Bearer " + TOKEN, "Accept": accept,
               "Content-Type": "application/json", "User-Agent": "ai-pr-reviewer",
               "Accept-Encoding": "identity"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def parse_diff(raw):
    """Turn a unified diff into a list of files with numbered lines."""
    files, cur, n = [], None, 0
    for line in raw.splitlines():
        if line.startswith("diff --git"):
            cur = None
            continue
        if line.startswith("+++ "):
            path = line[4:]
            if path == "/dev/null":
                cur = None
                continue
            if path.startswith("b/"):
                path = path[2:]
            if path.endswith(SKIP_SUFFIX) or path.split("/")[-1] in SKIP_NAMES:
                cur = None
                continue
            cur = {"path": path, "text": "", "valid": set()}
            files.append(cur)
            continue
        if cur is None:
            continue
        m = HUNK.match(line)
        if m:
            n = int(m.group(1))
            cur["text"] += line + "\n"
        elif line.startswith("+"):
            cur["text"] += "%d: %s\n" % (n, line)
            cur["valid"].add(n)
            n += 1
        elif line.startswith("-"):
            cur["text"] += "    " + line + "\n"
        elif line.startswith(" "):
            cur["text"] += "%d: %s\n" % (n, line)
            n += 1
    return files


def make_chunks(files):
    """Group files into pieces small enough for the free model."""
    chunks, buf = [], ""
    for f in files:
        block = "### FILE: %s\n%s\n" % (f["path"], f["text"])
        if len(block) > MAX_CHARS:
            block = block[:MAX_CHARS] + "\n[truncated]\n"
        if buf and len(buf) + len(block) > MAX_CHARS:
            chunks.append(buf)
            buf = ""
        buf += block
    if buf:
        chunks.append(buf)
    return chunks


def parse_answer(text):
    """Read the model's JSON answer; never crash on bad output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        return {"summary": text[:500], "comments": []}
    data.setdefault("summary", "")
    data.setdefault("comments", [])
    return data


def ask_model(diff_text):
    payload = {"model": MODEL, "temperature": 0.2, "max_tokens": 1500,
               "messages": [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": diff_text}]}
    for attempt in range(4):
        status, body = send("POST", LLM_URL, payload, accept="application/json")
        if status in (429, 500, 502, 503) or (status == 200 and not body.strip()):
            print("Model busy or empty reply (%d), retrying..." % status)
            time.sleep(2 * 2 ** attempt)
            continue
        if status != 200:
            print("Model error %d: %s" % (status, body[:500]))
            sys.exit(1)
        try:
            reply = json.loads(body)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            print("Unexpected reply from the model (status %d): %r" % (status, body[:500]))
            sys.exit(1)
        return parse_answer(reply)
    print("Model kept failing, giving up.")
    sys.exit(1)


def collect(raw_diff):
    """Returns (summaries, comments). Comments on lines not in the diff are dropped."""
    files = parse_diff(raw_diff)
    valid = {f["path"]: f["valid"] for f in files}
    summaries, comments = [], []
    for chunk in make_chunks(files):
        result = ask_model(chunk)
        if result["summary"]:
            summaries.append(result["summary"])
        for c in result["comments"]:
            if not isinstance(c, dict):
                continue
            path, line = c.get("path"), c.get("line")
            if path in valid and line in valid[path]:
                comments.append({"path": path, "line": line,
                                 "severity": c.get("severity", "low"),
                                 "comment": c.get("comment", "")})
    return summaries, comments


def main():
    if not (TOKEN and REPO and PR_NUMBER):
        print("Missing GITHUB_TOKEN, GITHUB_REPOSITORY or PR_NUMBER.")
        sys.exit(1)
    base = "https://api.github.com/repos/%s/pulls/%s" % (REPO, PR_NUMBER)

    status, diff = send("GET", base, accept="application/vnd.github.v3.diff")
    if status != 200:
        print("Could not read the pull request diff (%d): %s" % (status, diff[:500]))
        sys.exit(1)

    summaries, comments = collect(diff)
    body = "## 🤖 AI Review\n" + (" ".join(summaries) or "No summary.")
    body += "\n\n%d comment(s)." % len(comments)
    gh_comments = [{"path": c["path"], "line": c["line"], "side": "RIGHT",
                    "body": "%s %s" % (ICONS.get(c["severity"], "🔵"), c["comment"])}
                   for c in comments]

    status, resp = send("POST", base + "/reviews",
                        {"body": body, "event": "COMMENT", "comments": gh_comments})
    if status == 422 and gh_comments:  # a line was rejected: post everything as one comment
        extra = "\n".join("- `%s:%s` %s" % (c["path"], c["line"], c["body"]) for c in gh_comments)
        status, resp = send("POST", base + "/reviews",
                            {"body": body + "\n\n" + extra, "event": "COMMENT"})
    if status not in (200, 201):
        print("Could not post the review (%d): %s" % (status, resp[:500]))
        sys.exit(1)
    print("Review posted with %d comment(s)." % len(comments))


if __name__ == "__main__":
    main()
