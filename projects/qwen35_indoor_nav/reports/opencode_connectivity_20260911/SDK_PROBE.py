"""Bounded, temporary-key diagnostic; never edits OpenCode or system settings."""
from pathlib import Path
import datetime
import getpass
import json
import signal
import socket
import time

import httpx
import openai
from openai import OpenAI

HOST = "llm-5wkwvwzntbklitq4.cn-beijing.maas.aliyuncs.com"
BASE_URL = "https://" + HOST + "/compatible-mode/v1"
# Matched A records from independent queries to 223.5.5.5 and 119.29.29.29.
# Diagnostic only: not a permanent DNS configuration or production address pin.
VERIFIED_IP = "47.94.20.201"


def main():
    key = getpass.getpass("Temporary API key (hidden; not saved): ")
    original_gai = socket.getaddrinfo
    results = []
    for mode in ("official_sdk_existing_proxy", "official_sdk_direct_dns_override"):
        started = time.monotonic()
        result = {"mode": mode, "time": datetime.datetime.now().isoformat(),
                  "sdk_version": openai.__version__, "model": "kimi-k3",
                  "base_url": BASE_URL, "tls_verification": True,
                  "max_output_tokens": 256, "automatic_retries": 0}
        client = None
        completion = None
        try:
            signal.alarm(45)
            kwargs = {}
            if mode.endswith("direct_dns_override"):
                def resolved(host, port, *args, **kwargs):
                    return original_gai(VERIFIED_IP if host == HOST else host,
                                        port, *args, **kwargs)
                socket.getaddrinfo = resolved
                kwargs["http_client"] = httpx.Client(
                    trust_env=False, verify=True, follow_redirects=False,
                    timeout=httpx.Timeout(20.0, connect=8.0))
            client = OpenAI(api_key=key, base_url=BASE_URL,
                            max_retries=0, timeout=20.0, **kwargs)
            completion = client.chat.completions.create(
                model="kimi-k3", messages=[{"role": "user", "content": "你是谁"}],
                stream=True, max_tokens=256)
            result["http_status"] = completion.response.status_code
            result["request_id"] = (completion.response.headers.get("x-request-id")
                                    or completion.response.headers.get("x-dashscope-request-id"))
            answer = []
            reasoning_chars = 0
            chunks = 0
            finish_reasons = []
            for chunk in completion:
                chunks += 1
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        answer.append(delta.content)
                    reasoning_chars += len(getattr(delta, "reasoning_content", None) or "")
                    if choice.finish_reason:
                        finish_reasons.append(choice.finish_reason)
            result.update(answer="".join(answer), reasoning_chars=reasoning_chars,
                          chunks=chunks, finish_reasons=finish_reasons,
                          stream_iterator_completed=True)
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["http_status"] = getattr(exc, "status_code", None)
            # Do not dump SDK request objects, headers, tracebacks, or key material.
            causes = []
            current = exc.__cause__
            while current is not None and len(causes) < 4:
                causes.append(type(current).__name__)
                current = current.__cause__
            result["cause_types"] = causes
        finally:
            signal.alarm(0)
            if completion is not None:
                completion.close()
            if client is not None:
                client.close()
            socket.getaddrinfo = original_gai
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        safe = json.dumps(result, ensure_ascii=False).replace(key, "[REDACTED]")
        print(safe, flush=True)
        results.append(json.loads(safe))
    key = ""
    destination = Path(__file__).with_name("SDK_PROBE_RESULT.json")
    # Preserve an earlier diagnostic result if this script is run again.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
