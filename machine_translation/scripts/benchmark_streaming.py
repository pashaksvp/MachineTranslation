from __future__ import annotations

import argparse
import json
import statistics
import time
from urllib import request


def measure_once(url: str, text: str) -> tuple[float, float, int]:
    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    token_count = 0
    with request.urlopen(req, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if "token" in payload:
                token_count += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            if payload.get("done"):
                break
    end = time.perf_counter()
    ttft = (first_token_at or end) - start
    tokens_per_second = token_count / max(end - (first_token_at or start), 1e-9)
    return ttft, tokens_per_second, token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/translate")
    parser.add_argument("--text", default="šarrum ana ālim illik")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    measurements = [measure_once(args.url, args.text) for _ in range(args.runs)]
    ttfts = [x[0] for x in measurements]
    speeds = [x[1] for x in measurements]
    print(f"median TTFT: {statistics.median(ttfts):.4f}s")
    print(f"p95 TTFT: {sorted(ttfts)[max(0, int(0.95 * len(ttfts)) - 1)]:.4f}s")
    print(f"median tokens/sec: {statistics.median(speeds):.2f}")


if __name__ == "__main__":
    main()
