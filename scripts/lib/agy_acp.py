#!/usr/bin/env python3
"""agy_acp.py — headless Agent Client Protocol (ACP) client for `agy acp`.

Part of the "Antigravity for Claude Code" plugin. agy-delegate.sh runs one delegated
turn through this client instead of `agy --print` whenever the installed agy offers
`agy acp` (see TRANSPORT in agy-delegate.sh). It speaks JSON-RPC 2.0 over agy's stdio:

    initialize -> session/new | session/list + session/resume
               -> session/set_model, session/set_mode, session/set_config_option
               -> session/prompt  (streams session/update, answers session/request_permission)
               -> session/close

and prints ONE JSON envelope on stdout in the same shape `agy --print --output-format
json` emits, so the wrapper's envelope handling, failure classification, model fallback,
AGY_USAGE line and exit-code contract stay transport-independent:

    {"conversation_id", "status": "SUCCESS"|"ERROR", "response", "error",
     "duration_seconds", "num_turns", "usage": {"input_tokens", "output_tokens",
     "thinking_tokens", "cache_read_tokens", "total_tokens"},
     "denied_actions": [{"action", "display_name"}], "finished_reason"?, "transport": "acp",
     "tool_calls": {"total", "failed"}}

Exit codes mirror agy 1.5.x print mode: 0 SUCCESS | 1 ERROR | 3 print_timeout (the turn
was cancelled at --timeout; the partial reply is in "response").

What ACP buys over print mode:
  * Permission requests reach the CLIENT. Headless print mode can only soft-deny them or
    approve everything (--yolo). Here each request is answered on its own: --approve
    <kinds> allows just those ACP tool kinds (edit, execute, fetch, ...), everything else
    is denied and named in denied_actions.
  * A timeout is a clean `session/cancel` (which also stops the turn's subagents and
    background tasks) followed by `session/close`, not a SIGTERM to a hung process.
  * Prompts travel over a pipe, so there is no argv / E2BIG ceiling on their size.
  * Model availability is read from the session (`models.availableModels`) instead of
    being scraped from an error message.

Standard library only; Python 3.6+ (macOS system python3 is the floor).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

try:
    import queue
except ImportError:  # pragma: no cover - py2 is not supported, but fail readably
    sys.stderr.write("agy_acp: python3 is required\n")
    sys.exit(1)

PROTOCOL_VERSION = 1
CLIENT_INFO = {"name": "antigravity-for-claude-code", "version": "acp-transport"}
EFFORT_ORDER = ["low", "medium", "high", "xhigh", "max"]
# Grace periods (seconds). CANCEL_GRACE: how long a cancelled turn may take to report
# `stopReason: cancelled`. SHUTDOWN_GRACE: agy 1.5.5+ bounds background-task shutdown
# at 30 s; give it that plus a little before escalating to SIGTERM/SIGKILL.
CANCEL_GRACE = float(os.environ.get("AGY_ACP_CANCEL_GRACE", "10"))
SHUTDOWN_GRACE = float(os.environ.get("AGY_ACP_SHUTDOWN_GRACE", "35"))
SETUP_TIMEOUT = float(os.environ.get("AGY_ACP_SETUP_TIMEOUT", "120"))


def warn(msg):
    sys.stderr.write("agy-acp: %s\n" % msg)
    sys.stderr.flush()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def parse_duration(d):
    """'5m' / '300s' / '1h' / '300' -> seconds (float). 0 = unbounded."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", str(d or ""))
    if not m:
        raise ValueError("invalid --timeout %r (use e.g. 300s, 5m, 1h, or 0)" % d)
    n = float(m.group(1))
    return n * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


class Envelope(object):
    def __init__(self):
        self.started = time.time()
        self.conversation_id = ""
        self.response = []
        self.error = ""
        self.finished_reason = ""
        self.usage = {}
        self.denied = []
        self.tool_calls = {}
        self.num_turns = 0

    def to_json(self, status):
        u = self.usage or {}

        def n(k):
            v = u.get(k)
            return v if isinstance(v, int) and not isinstance(v, bool) else 0

        failed = sum(1 for s in self.tool_calls.values() if s == "failed")
        d = {
            "conversation_id": self.conversation_id,
            "status": status,
            "response": "".join(self.response),
            "duration_seconds": round(time.time() - self.started, 3),
            "num_turns": self.num_turns,
            "usage": {
                "input_tokens": n("inputTokens"),
                "output_tokens": n("outputTokens"),
                "thinking_tokens": n("thoughtTokens"),
                "cache_read_tokens": n("cachedReadTokens"),
                "total_tokens": n("totalTokens"),
            },
            "transport": "acp",
            "tool_calls": {"total": len(self.tool_calls), "failed": failed},
        }
        if self.error:
            d["error"] = self.error
        if self.denied:
            d["denied_actions"] = self.denied
        if self.finished_reason:
            d["finished_reason"] = self.finished_reason
        return d


class RpcError(Exception):
    def __init__(self, err):
        self.err = err if isinstance(err, dict) else {"message": str(err)}
        Exception.__init__(self, self.describe())

    def describe(self):
        msg = str(self.err.get("message", "") or "JSON-RPC error")
        data = self.err.get("data")
        if isinstance(data, str) and data.strip() and data.strip() not in msg:
            msg = "%s: %s" % (msg, data.strip())
        elif isinstance(data, dict):
            detail = data.get("details") or data.get("message") or data.get("reason")
            if detail and str(detail) not in msg:
                msg = "%s: %s" % (msg, detail)
        return msg


class AgentExited(Exception):
    pass


class AcpClient(object):
    def __init__(self, argv, policy, env):
        self.policy = policy
        self.env = env
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        )
        self.inbox = queue.Queue()
        self.next_id = 0
        self.wlock = threading.Lock()
        self.prompt_session = None
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # --- transport ---------------------------------------------------------
    def _reader(self):
        # One JSON-RPC message per line (ACP stdio framing).
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.strip()
            if not line:
                continue
            try:
                self.inbox.put(json.loads(line.decode("utf-8", "replace")))
            except ValueError:
                warn("ignoring non-JSON line from agy acp: %r" % line[:200])
        self.inbox.put(None)

    def _write(self, obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with self.wlock:
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                raise AgentExited("agy acp closed its input")

    def notify(self, method, params):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def send(self, method, params):
        self.next_id += 1
        self._write({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        return self.next_id

    def pump(self, timeout):
        """Handle one inbound message. Returns the message, or None on timeout."""
        try:
            msg = self.inbox.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if msg is None:
            self.inbox.put(None)  # keep EOF sticky for later callers
            rc = self.proc.poll()
            raise AgentExited("agy acp exited%s" % ("" if rc is None else " (rc %s)" % rc))
        if "method" in msg:
            if "id" in msg:
                self._on_request(msg)
            else:
                self._on_notification(msg)
        return msg

    def wait_for(self, req_id, timeout):
        """Pump until the response to req_id arrives. Returns its result or raises."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            left = 3600.0 if deadline is None else deadline - time.time()
            if left <= 0:
                return None
            msg = self.pump(min(left, 1.0))
            if msg is None or "method" in msg or msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise RpcError(msg["error"])
            return msg.get("result") or {}

    def call(self, method, params, timeout=SETUP_TIMEOUT):
        rid = self.send(method, params)
        res = self.wait_for(rid, timeout)
        if res is None:
            raise RpcError({"message": "timed out waiting for %s" % method})
        return res

    # --- inbound -----------------------------------------------------------
    def _on_notification(self, msg):
        if msg.get("method") != "session/update":
            return
        p = msg.get("params") or {}
        if self.prompt_session is None or p.get("sessionId") != self.prompt_session:
            return  # history replay (session/load) or another session: not this turn
        upd = p.get("update") or {}
        kind = upd.get("sessionUpdate")
        env = self.env
        if kind == "agent_message_chunk":
            c = upd.get("content") or {}
            if c.get("type") == "text":
                env.response.append(str(c.get("text", "")))
        elif kind in ("tool_call", "tool_call_update"):
            tid = upd.get("toolCallId")
            if tid:
                env.tool_calls[tid] = upd.get("status") or env.tool_calls.get(tid, "")

    def _on_request(self, msg):
        method = msg.get("method")
        if method == "session/request_permission":
            self._write({"jsonrpc": "2.0", "id": msg["id"],
                         "result": {"outcome": self.policy.decide(msg.get("params") or {}, self.env)}})
        else:
            # We advertise no fs/terminal capabilities, so anything else is unexpected.
            self._write({"jsonrpc": "2.0", "id": msg["id"],
                         "error": {"code": -32601, "message": "method not supported by headless client: %s" % method}})

    # --- shutdown ----------------------------------------------------------
    def shutdown(self, session_id):
        if session_id and self.proc.poll() is None:
            try:
                rid = self.send("session/close", {"sessionId": session_id})
                self.wait_for(rid, 10)
            except Exception:
                pass
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=SHUTDOWN_GRACE)
        except subprocess.TimeoutExpired:
            warn("agy acp did not exit %ss after the session closed; terminating it" % int(SHUTDOWN_GRACE))
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class PermissionPolicy(object):
    """Answers session/request_permission. Denials are recorded for denied_actions."""

    def __init__(self, yolo, approve):
        self.yolo = yolo
        self.approve = set(a for a in approve if a)

    def decide(self, params, env):
        tc = params.get("toolCall") or {}
        opts = [o for o in (params.get("options") or []) if isinstance(o, dict)]
        kind = str(tc.get("kind") or "")
        name = str(tc.get("name") or tc.get("title") or kind or "tool")
        allowed = self.yolo or "all" in self.approve or kind in self.approve or name in self.approve
        want = ("allow_once", "allow_always") if allowed else ("reject_once", "reject_always")
        pick = None
        for k in want:
            pick = next((o for o in opts if o.get("kind") == k), None)
            if pick:
                break
        if not allowed:
            label = str(tc.get("title") or kind or name)
            entry = {"action": name, "display_name": label}
            if entry not in env.denied:
                env.denied.append(entry)
        if pick is None:
            return {"outcome": "cancelled"}
        return {"outcome": "selected", "optionId": pick.get("optionId")}


def resolve_model(requested, models):
    """Map a display name / slug to a modelId from session/new's availableModels."""
    avail = (models or {}).get("availableModels") or []
    if not requested:
        return None, avail
    rn = norm(requested)
    for m in avail:
        if requested in (m.get("modelId"), m.get("name")):
            return m.get("modelId"), avail
    for m in avail:
        if rn and rn in (norm(m.get("modelId")), norm(m.get("name"))):
            return m.get("modelId"), avail
    return None, avail


def unknown_model_error(requested, avail):
    # Same wording and `Available models:` block shape as agy's print-mode
    # ValidateRequestedModel error, so model_unavailable_blob() and
    # pick_available_fallback() in agy-delegate.sh work unchanged (one-hop fallback).
    lines = ['invalid model selection (--model "%s"): model %s is not recognized as a known model'
             % (requested, requested), "Available models:"]
    for m in avail:
        lines.append("  %s  %s" % (m.get("modelId", ""), m.get("name", "")))
    return "\n".join(lines)


def thought_option(config_options):
    for o in config_options or []:
        if isinstance(o, dict) and (o.get("id") == "thought_level" or o.get("category") == "thought_level"):
            vals = [str(v.get("value")) for v in (o.get("options") or []) if isinstance(v, dict)]
            return o.get("id"), vals
    return None, []


def pick_effort(effort, values):
    if effort in values:
        return effort
    if not values:
        return None
    # Clamp to the closest level the model offers (e.g. xhigh/max -> high).
    want = EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else 2
    ranked = [v for v in EFFORT_ORDER if v in values]
    if not ranked:
        return None
    best = ranked[0]
    for v in ranked:
        if EFFORT_ORDER.index(v) <= want:
            best = v
    return best


def parse_args(argv):
    a = {"model": "", "effort": "", "add_dirs": [], "yolo": False, "mode": "", "sandbox": False,
         "continue": False, "conversation": "", "timeout": "5m", "approve": [], "prompt_file": "",
         "agy": os.environ.get("AGY_BIN", "agy")}
    i = 0
    while i < len(argv):
        k = argv[i]

        def val():
            if i + 1 >= len(argv):
                raise SystemExit("agy-acp: option %s needs a value" % k)
            return argv[i + 1]
        if k in ("--model", "--effort", "--mode", "--conversation", "--timeout", "--prompt-file", "--agy"):
            a[k[2:].replace("-", "_")] = val(); i += 2
        elif k == "--add-dir":
            a["add_dirs"].append(val()); i += 2
        elif k == "--approve":
            a["approve"] += [x.strip().lower() for x in val().split(",") if x.strip()]; i += 2
        elif k == "--yolo":
            a["yolo"] = True; i += 1
        elif k == "--sandbox":
            a["sandbox"] = True; i += 1
        elif k == "--continue":
            a["continue"] = True; i += 1
        else:
            raise SystemExit("agy-acp: unknown option %s" % k)
    if not a["prompt_file"]:
        raise SystemExit("agy-acp: --prompt-file is required")
    return a


def run(a):
    env = Envelope()
    try:
        timeout = parse_duration(a["timeout"])
    except ValueError as e:
        raise SystemExit("agy-acp: %s" % e)
    with open(a["prompt_file"], "rb") as fh:
        prompt = fh.read().decode("utf-8", "replace")

    argv = [a["agy"], "acp"]
    if a["yolo"]:
        argv.append("--dangerously-skip-permissions")
    if a["sandbox"]:
        argv.append("--sandbox")
    cwd = os.getcwd()
    add_dirs = [os.path.abspath(d) for d in a["add_dirs"]]

    try:
        client = AcpClient(argv, PermissionPolicy(a["yolo"], a["approve"]), env)
    except OSError as e:
        env.error = "could not start `agy acp`: %s" % e
        return env, "ERROR", 1

    sid = None
    status, rc = "ERROR", 1
    try:
        client.call("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": CLIENT_INFO,
        })
        session_params = {"cwd": cwd, "mcpServers": []}
        if add_dirs:
            session_params["additionalDirectories"] = add_dirs

        sess = None
        resume_id = a["conversation"]
        if a["continue"] and not resume_id:
            listed = client.call("session/list", {"cwd": cwd}).get("sessions") or []
            listed = [s for s in listed if isinstance(s, dict) and s.get("sessionId")]
            listed.sort(key=lambda s: str(s.get("updatedAt", "")), reverse=True)
            if listed:
                resume_id = listed[0]["sessionId"]
            else:
                warn("--continue: no earlier conversation in %s; starting a new one" % cwd)
        if resume_id:
            p = dict(session_params, sessionId=resume_id)
            sess = client.call("session/resume", p)
            sid = sess.get("sessionId") or resume_id
        else:
            sess = client.call("session/new", session_params)
            sid = sess.get("sessionId")
        if not sid:
            raise RpcError({"message": "agy acp returned no sessionId"})
        env.conversation_id = sid

        if a["model"]:
            model_id, avail = resolve_model(a["model"], sess.get("models"))
            if model_id is None:
                env.error = unknown_model_error(a["model"], avail)
                return env, "ERROR", 1
            client.call("session/set_model", {"sessionId": sid, "modelId": model_id})

        if a["effort"]:
            opt_id, values = thought_option(sess.get("configOptions"))
            chosen = pick_effort(a["effort"], values)
            if opt_id and chosen:
                if chosen != a["effort"]:
                    warn("effort %s is not offered for this model; using %s" % (a["effort"], chosen))
                client.call("session/set_config_option", {"sessionId": sid, "configId": opt_id, "value": chosen})
            else:
                warn("this session offers no thinking-level option; ignoring --effort %s" % a["effort"])

        if a["mode"]:
            client.call("session/set_mode", {"sessionId": sid, "modeId": a["mode"]})

        # --- the turn --------------------------------------------------------
        client.prompt_session = sid
        env.num_turns = 1
        rid = client.send("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": prompt}]})
        res = client.wait_for(rid, timeout if timeout > 0 else None)
        if res is None:
            # --timeout expired mid-turn: cancel it (this also stops its subagents and
            # background tasks) and keep whatever the agent had said so far.
            client.notify("session/cancel", {"sessionId": sid})
            try:
                res = client.wait_for(rid, CANCEL_GRACE)
            except RpcError:
                res = None
            if res and isinstance(res.get("usage"), dict):
                env.usage = res["usage"]
            env.finished_reason = "print_timeout"
            env.error = "print timeout after %s with turn in progress; returning partial output" % a["timeout"]
            status, rc = "ERROR", 3
        else:
            env.usage = res.get("usage") if isinstance(res.get("usage"), dict) else {}
            stop = str(res.get("stopReason") or "")
            if stop in ("end_turn", ""):
                status, rc = "SUCCESS", 0
            elif stop in ("max_tokens", "max_turn_requests"):
                env.finished_reason = stop
                status, rc = "SUCCESS", 0
            elif stop == "cancelled":
                env.finished_reason = "cancelled"
                env.error = "the turn was cancelled by agy"
            else:
                env.finished_reason = stop
                env.error = "turn ended with stopReason %s" % stop
    except RpcError as e:
        env.error = e.describe()
    except AgentExited as e:
        env.error = "%s before the turn finished" % e
    finally:
        client.prompt_session = None
        client.shutdown(sid)
    return env, status, rc


def main():
    a = parse_args(sys.argv[1:])
    env, status, rc = run(a)
    sys.stdout.write(json.dumps(env.to_json(status), ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
