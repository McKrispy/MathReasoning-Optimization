"""EXP-R01 step 2 (env mp1ref): LLM-as-a-judge over the validation split.

Reference only: violates the official resource limits, never used by submission
code. Reads candidate texts (no reference answers), sends one structured-output
request per question and appends raw responses to a JSONL cache, one file per
backend and model. Cached questions are skipped, so reruns cost nothing.
Spending stops at the budget.

Backends:
  gemini  Google Gemini API, key from GEMINI_API_KEY.
  openai  any OpenAI-compatible Chat Completions endpoint, key from OPENAI_API_KEY,
          optional OPENAI_BASE_URL for third-party compatible services.
Keys are read from the environment only and never written anywhere.
"""
import argparse
import copy
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

WORKSPACE = Path(__file__).resolve().parents[3]
BANK_PATH = WORKSPACE / "DASE7506-MP2" / "code" / "data" / "candidates" / "gsm8k_validation.jsonl"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_v1.txt"
REF_RUNS = WORKSPACE / "dev" / "runs" / "ref"
ORDER_PATH = REF_RUNS / "validation_order.json"

GEMINI_PRICES_PER_MILLION = {  # USD, paid tier standard, prompts <= 200k tokens, checked 2026-10-19
    "gemini-4-argon": {"input": 2.00, "output": 10.00},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}
BUDGET_USD = 10.0
MAX_VALID_ATTEMPTS = 3
MAX_TRANSIENT_RETRIES = 6

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "own_answer": {"type": "string"},
        "candidates": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "final_answer": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["correct", "incorrect", "uncertain"]},
                    "first_error": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["index", "final_answer", "verdict", "first_error", "confidence"],
            },
        },
        "chosen_index": {"type": "integer", "minimum": 0, "maximum": 3},
        "chosen_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["own_answer", "candidates", "chosen_index", "chosen_confidence"],
}


class QuotaExhausted(RuntimeError):
    """Quota or balance used up; retrying in this run is pointless."""


def strict_schema(schema: dict) -> dict:
    """OpenAI strict mode: every object closed, numeric / length bounds dropped (checked in validate)."""
    out = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            for key in ("minimum", "maximum", "minItems", "maxItems"):
                node.pop(key, None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(template: str, row: dict) -> str:
    text = template.replace("{question}", row["question"])
    for cand in row["candidates"]:
        text = text.replace("{candidate_%d}" % cand["index"], cand["text"])
    return text


def validate(text: str):
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(parsed, dict) or parsed.get("chosen_index") not in (0, 1, 2, 3):
        return None, "bad_chosen_index"
    return parsed, None


class GeminiBackend:
    name = "gemini"

    def __init__(self, model: str, tier: str):
        from google import genai
        from google.genai import types

        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY is not set")
        self.model, self.tier = model, tier
        self.price = GEMINI_PRICES_PER_MILLION[model]
        self.client = genai.Client()
        self.config = types.GenerateContentConfig(response_mime_type="application/json",
                                                  response_json_schema=RESPONSE_SCHEMA)
        self.endpoint = "generativelanguage.googleapis.com"

    def call(self, prompt: str):
        for attempt in range(MAX_TRANSIENT_RETRIES):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
                meta = response.usage_metadata
                usage = {"prompt": meta.prompt_token_count or 0, "candidates": meta.candidates_token_count or 0,
                         "thoughts": meta.thoughts_token_count or 0}
                return response.text, usage, response.model_version
            except Exception as error:  # noqa: BLE001 - SDK raises several transport / API types
                code = getattr(error, "code", None)
                message = str(error)
                if code == 429 and ("PerDay" in message or "limit: 0" in message):
                    raise QuotaExhausted(message.split("\n")[0][:200]) from None
                if code not in (429, 500, 502, 503, 504, None) or attempt == MAX_TRANSIENT_RETRIES - 1:
                    raise
                delay = re.search(r"retryDelay'?: '(\d+)s", message)
                wait = int(delay.group(1)) + 1 if delay else 2 ** attempt + random.random()
                time.sleep(min(90, wait))

    def cost(self, usage: dict) -> float:
        if self.tier == "free":
            return 0.0
        return self.paid_equivalent(usage)

    def paid_equivalent(self, usage: dict) -> float:
        output = usage["candidates"] + usage["thoughts"]
        return (usage["prompt"] * self.price["input"] + output * self.price["output"]) / 1e6


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str, price_in: float, price_out: float, response_format: str,
                 reasoning_effort: str | None):
        import openai

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        self.openai = openai
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self.client = openai.OpenAI(base_url=base_url, max_retries=0, timeout=300)
        self.endpoint = urlparse(base_url).netloc if base_url else "api.openai.com"
        self.model, self.tier = model, "paid"
        self.price = {"input": price_in, "output": price_out}
        if response_format == "json_schema":
            self.response_format = {"type": "json_schema", "json_schema": {
                "name": "judge", "strict": True, "schema": strict_schema(RESPONSE_SCHEMA)}}
        else:
            self.response_format = {"type": "json_object"}
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def call(self, prompt: str):
        for attempt in range(MAX_TRANSIENT_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    response_format=self.response_format, **self.extra)
                usage = response.usage
                details = getattr(usage, "completion_tokens_details", None)
                reasoning = (getattr(details, "reasoning_tokens", None) or 0) if details else 0
                completion = usage.completion_tokens or 0
                tokens = {"prompt": usage.prompt_tokens or 0, "candidates": max(0, completion - reasoning),
                          "thoughts": reasoning}
                return response.choices[0].message.content, tokens, response.model
            except (self.openai.AuthenticationError, self.openai.PermissionDeniedError) as error:
                raise SystemExit(f"OpenAI rejected the key or model access (HTTP {error.status_code}); "
                                 "SDK message suppressed because it echoes part of the key") from None
            except self.openai.RateLimitError as error:
                if "insufficient_quota" in str(error):
                    raise QuotaExhausted(str(error)[:200]) from None
                if attempt == MAX_TRANSIENT_RETRIES - 1:
                    raise
                time.sleep(min(90, 2 ** attempt * 5 + random.random()))
            except (self.openai.APIConnectionError, self.openai.APITimeoutError, self.openai.InternalServerError):
                if attempt == MAX_TRANSIENT_RETRIES - 1:
                    raise
                time.sleep(min(90, 2 ** attempt * 2 + random.random()))

    def cost(self, usage: dict) -> float:
        return self.paid_equivalent(usage)

    def paid_equivalent(self, usage: dict) -> float:
        output = usage["candidates"] + usage["thoughts"]
        return (usage["prompt"] * self.price["input"] + output * self.price["output"]) / 1e6


def judge(backend, template: str, row: dict) -> dict:
    prompt = build_prompt(template, row)
    attempts, totals = [], {"prompt": 0, "candidates": 0, "thoughts": 0}
    parsed = None
    tick = time.perf_counter()
    for _ in range(MAX_VALID_ATTEMPTS):
        text, usage, version = backend.call(prompt)
        for key in totals:
            totals[key] += usage[key]
        parsed, error = validate(text)
        attempts.append({"usage": usage, "error": error, "model_version": version, "raw_text": text})
        if parsed is not None:
            break
    return {"id": row["id"], "backend": backend.name, "endpoint": backend.endpoint, "model": backend.model,
            "tier": backend.tier, "prompt_file": PROMPT_PATH.name,
            "called_at": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - tick,
            "reasoning_effort": getattr(backend, "extra", {}).get("reasoning_effort", "default"),
            "attempts": attempts, "usage_total": totals, "price_per_million": backend.price,
            "cost_usd": backend.cost(totals), "paid_equivalent_usd": backend.paid_equivalent(totals),
            "parsed": parsed, "failed": parsed is None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gemini", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tier", choices=("free", "paid"), default="paid", help="gemini only")
    parser.add_argument("--price-in", type=float, help="openai: USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, help="openai: USD per 1M output tokens, reasoning included")
    parser.add_argument("--response-format", choices=("json_schema", "json_object"), default="json_schema")
    parser.add_argument("--reasoning-effort", default=None, help="openai reasoning models only")
    parser.add_argument("--limit", type=int, default=None, help="questions to send in this run")
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    args = parser.parse_args()

    if args.backend == "gemini":
        if args.model not in GEMINI_PRICES_PER_MILLION:
            raise SystemExit(f"unknown gemini model {args.model}")
        backend = GeminiBackend(args.model, args.tier)
    else:
        if args.price_in is None or args.price_out is None:
            raise SystemExit("--price-in and --price-out are required for the openai backend")
        backend = OpenAIBackend(args.model, args.price_in, args.price_out, args.response_format,
                                args.reasoning_effort)

    template = PROMPT_PATH.read_text(encoding="utf-8")
    bank = {row["id"]: row for row in load_jsonl(BANK_PATH)}
    order = json.loads(ORDER_PATH.read_text(encoding="utf-8"))["questions"]
    ordered = [q["id"] for q in order if q["hard"]] + ([] if args.hard_only else [q["id"] for q in order if not q["hard"]])

    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    if args.backend == "openai" and args.reasoning_effort:
        safe_model += f"_effort-{args.reasoning_effort}"
    cache_path = REF_RUNS / f"exp_r01_judge_cache_{args.backend}_{safe_model}.jsonl"
    cached = load_jsonl(cache_path)
    done = {entry["id"] for entry in cached}
    spent = sum(entry["cost_usd"] for entry in cached)
    todo = [qid for qid in ordered if qid not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{args.backend}/{args.model} @ {backend.endpoint} | cached {len(done)} | spent ${spent:.4f} "
          f"| to send {len(todo)} | budget ${args.budget:.2f}")

    lock = threading.Lock()
    state = {"spent": spent, "stopped": None, "sent": 0, "run_cost": 0.0}

    def work(qid: str):
        with lock:
            if state["stopped"]:
                return None
            if state["spent"] >= args.budget:
                state["stopped"] = "budget"
                return None
        try:
            entry = judge(backend, template, bank[qid])
        except QuotaExhausted as error:
            with lock:
                if not state["stopped"]:
                    state["stopped"] = "quota"
                    print(f"  quota exhausted, stopping: {error}", flush=True)
            return None
        with lock:
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            state["spent"] += entry["cost_usd"]
            state["run_cost"] += entry["paid_equivalent_usd"]
            state["sent"] += 1
            if state["sent"] % 10 == 0 or state["sent"] == len(todo):
                print(f"  {state['sent']}/{len(todo)} | spent ${state['spent']:.4f}", flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, qid) for qid in todo]):
            future.result()
    print(f"done | sent {state['sent']} | total spent ${state['spent']:.4f} | stopped by {state['stopped']}")
    if state["sent"]:
        per_question = state["run_cost"] / state["sent"]
        remaining = len([qid for qid in ordered if qid not in done]) - state["sent"]
        print(f"estimate | ${per_question:.5f} per question | full {len(ordered)} questions "
              f"${per_question * len(ordered):.2f} | remaining {remaining} questions ${per_question * remaining:.2f}")


if __name__ == "__main__":
    main()
