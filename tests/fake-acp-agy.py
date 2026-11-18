#!/usr/bin/env python3
"""Fake `agy` for the ACP transport tests in run-tests.sh (no network, no real agy).

    agy --help   -> advertises the `acp` subcommand (plus the print-mode flags)
    agy models   -> lists STUB_ACP_MODELS
    agy acp ...  -> a scripted ACP server on stdio, steered by STUB_ACP_MODE:

      ok          reply with the session state: model, mode, effort, dirs, how it started
      len         reply LEN=<bytes of the prompt> (large-prompt transport)
      permission  ask for an `execute` permission (run_command); reply RAN only if allowed
      quota       session/prompt fails with a quota error
      auth        session/new fails with "Authentication required"
      slow        stream PARTIAL, then wait for session/cancel -> stopReason "cancelled"
      linger      like ok, but ignore stdin EOF for a while (shutdown escalation)

Every request/notification it receives is appended to $STUB_ACP_LOG (one
`<method> <json params>` line), with an `argv <args>` line first, so tests can assert
what the client sent.
"""
import json
import os
import sys
import time

MODELS = [m for m in os.environ.get(
    "STUB_ACP_MODELS",
    "Gemini 4 Hydrogen (High)|Gemini 4 Hydrogen (Low)|Gemini 4 Argon (High)|Gemini 3.8 Flash (High)",
).split("|") if m]
MODE = os.environ.get("STUB_ACP_MODE", "ok")
LOG = os.environ.get("STUB_ACP_LOG", "")


def slug(name):
    return name.lower().replace("(", "").replace(")", "").replace(" ", "-")


def log(line):
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def update(sid, upd):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": upd}})


def read():
    line = sys.stdin.readline()
    if not line:
        return None
    msg = json.loads(line)
    if "method" in msg:
        log("%s %s" % (msg["method"], json.dumps(msg.get("params", {}), sort_keys=True)))
    return msg


def session_payload(sid):
    return {
        "sessionId": sid,
        "models": {"availableModels": [{"modelId": slug(m), "name": m} for m in MODELS],
                   "currentModelId": slug(MODELS[0]) if MODELS else ""},
        "modes": {"currentModeId": "default", "availableModes": [
            {"id": i, "name": i} for i in ("default", "accept-edits", "auto", "plan")]},
        "configOptions": [{"id": "thought_level", "category": "thought_level", "type": "select",
                           "currentValue": "medium",
                           "options": [{"value": v, "name": v} for v in ("low", "medium", "high")]}],
    }


def serve():
    state = {"model": "", "mode": "default", "effort": "", "dirs": 0, "start": ""}
    sid = None
    while True:
        msg = read()
        if msg is None:
            if MODE == "linger":
                time.sleep(float(os.environ.get("STUB_ACP_LINGER", "30")))
            return
        m, mid, p = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if m is None:
            continue  # a response to our own request is handled inline below
        if m == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1, "agentCapabilities": {
                "loadSession": True, "sessionCapabilities": {"resume": {}, "list": {}, "close": {}}}}})
        elif m == "session/list":
            raw = os.environ.get("STUB_ACP_SESSIONS")
            sessions = json.loads(raw) if raw else [
                {"sessionId": "older-1", "cwd": p.get("cwd"), "updatedAt": "2026-01-01T00:00:00Z"},
                {"sessionId": "prev-1", "cwd": p.get("cwd"), "updatedAt": "2026-02-01T00:00:00Z"}]
            send({"jsonrpc": "2.0", "id": mid, "result": {"sessions": sessions}})
        elif m in ("session/new", "session/resume", "session/load"):
            if MODE == "auth":
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": "Authentication required"}})
                continue
            sid = p.get("sessionId") or "stub-session-1"
            state["start"] = "new" if m == "session/new" else "resumed:%s" % sid
            state["dirs"] = len(p.get("additionalDirectories") or [])
            send({"jsonrpc": "2.0", "id": mid, "result": session_payload(sid)})
        elif m == "session/set_model":
            if p.get("modelId") not in [slug(x) for x in MODELS]:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Unknown model ID: %s" % p.get("modelId")}})
            else:
                state["model"] = p["modelId"]
                send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif m == "session/set_mode":
            state["mode"] = p.get("modeId")
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif m == "session/set_config_option":
            state["effort"] = p.get("value")
            send({"jsonrpc": "2.0", "id": mid, "result": {"configOptions": []}})
        elif m == "session/close":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif m == "session/cancel":
            pass
        elif m == "session/prompt":
            prompt(mid, sid, p, state)
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no " + m}})


USAGE = {"inputTokens": 100, "outputTokens": 20, "thoughtTokens": 5, "cachedReadTokens": 40, "totalTokens": 160}


def prompt(mid, sid, p, state):
    text = "".join(b.get("text", "") for b in p.get("prompt") or [] if isinstance(b, dict))
    if MODE == "quota":
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000,
              "message": "RESOURCE_EXHAUSTED: quota exceeded for this model"}})
        return
    if MODE == "permission":
        update(sid, {"sessionUpdate": "tool_call", "toolCallId": "t1", "name": "run_command",
                     "kind": "execute", "title": "uname -s", "status": "pending"})
        send({"jsonrpc": "2.0", "id": "perm-1", "method": "session/request_permission", "params": {
            "sessionId": sid,
            "toolCall": {"toolCallId": "t1", "name": "run_command", "kind": "execute", "title": "uname -s"},
            "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "deny", "name": "Deny", "kind": "reject_once"}]}})
        reply = None
        while reply is None:
            msg = read()
            if msg is None:
                return
            if msg.get("id") == "perm-1" and "method" not in msg:
                reply = msg
        log("permission-answer %s" % json.dumps(reply.get("result"), sort_keys=True))
        chosen = ((reply.get("result") or {}).get("outcome") or {}).get("optionId")
        update(sid, {"sessionUpdate": "tool_call_update", "toolCallId": "t1",
                     "status": "completed" if chosen == "allow" else "failed"})
        if chosen == "allow":
            update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "RAN"}})
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn", "usage": USAGE}})
        return
    if MODE == "slow":
        update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "PARTIAL"}})
        while True:
            msg = read()
            if msg is None:
                return
            if msg.get("method") == "session/cancel":
                send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "cancelled", "usage": USAGE}})
                return
    if MODE == "len":
        out = "LEN=%d" % len(text.encode("utf-8"))
    else:
        out = "ACP_OK model=%s mode=%s effort=%s dirs=%d start=%s" % (
            state["model"], state["mode"], state["effort"] or "-", state["dirs"], state["start"])
    # Stream in two chunks to exercise reassembly.
    half = len(out) // 2
    update(sid, {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking..."}})
    update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": out[:half]}})
    update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": out[half:]}})
    send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn", "usage": USAGE}})


def main():
    args = sys.argv[1:]
    if args[:1] in (["--help"], ["-h"]):
        print("Usage of agy:\n  --output-format  Output format for print mode (text, json, stream-json)\n"
              "  --disable-slash-commands  Disable slash command expansion in print mode\n"
              "  --settings  Settings for this session only\n\n"
              "Available subcommands:\n  acp             Start the Agent Client Protocol (ACP) stdio server\n"
              "  models          List available models")
        return 0
    if args[:1] == ["models"]:
        print("\n".join(slug(m) for m in MODELS))
        return 0
    if args[:1] == ["acp"]:
        log("argv %s" % " ".join(args))
        serve()
        return 0
    # Any print-mode call means the wrapper picked the wrong transport.
    print("FAKE_ACP_AGY_PRINT_MODE %s" % " ".join(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
