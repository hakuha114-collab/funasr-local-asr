# -*- encoding: utf-8 -*-
"""
Client example: speaker diarization request.

This shows how to call the local diarization HTTP service (the cam++ real
clustering path) started by `funasr_local_server/start_diarize_server.bat`
on port 10098, and print speaker-separated sentences with timestamps.

Start the service first:
    python funasr_local_server/diarize_server.py --port 10098 --device cpu

Then run this client:
    python diarize_client.py ../audio/meeting.wav
    python diarize_client.py talk.mp3 --max_speakers 3 --itn 1 --hotword "阿里巴巴 20"

Dependency: requests  (pip install requests)
"""
import argparse
import json
import os
import sys

import requests


def parse_args():
    p = argparse.ArgumentParser(description="FunASR diarization client example (port 10098)")
    p.add_argument("audio", help="audio file path (wav/flac/ogg recommended; mp3 etc. need ffmpeg)")
    p.add_argument("--host", default="localhost", help="diarize server host (default: localhost)")
    p.add_argument("--port", type=int, default=10098, help="diarize server port (default: 10098)")
    p.add_argument("--itn", type=int, default=0, help="1 = inverse text normalization")
    p.add_argument("--hotword", default="", help='space-separated hotwords, e.g. "阿里巴巴 20"')
    p.add_argument("--max_speakers", type=int, default=0, help="speaker cap; 0 = auto")
    p.add_argument("--timeout", type=int, default=600, help="request timeout seconds")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.audio):
        print("ERROR: audio file not found: %s" % args.audio, file=sys.stderr)
        sys.exit(2)

    base = "http://%s:%d" % (args.host, args.port)

    # 1) health check —— 给出可读的连接失败提示
    try:
        r = requests.get(base + "/health", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("ERROR: cannot connect to %s" % base, file=sys.stderr)
        print("  is the diarization service running? start it with:", file=sys.stderr)
        print("    python funasr_local_server/diarize_server.py --port %d --device cpu"
              % args.port, file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print("ERROR: health check failed: %s" % e, file=sys.stderr)
        sys.exit(1)

    # 2) POST /diarize (multipart form)
    url = base + "/diarize"
    data = {"itn": args.itn, "max_speakers": args.max_speakers}
    if args.hotword.strip():
        data["hotword"] = args.hotword.strip()

    print("POST %s  file=%s" % (url, args.audio))
    try:
        with open(args.audio, "rb") as f:
            files = {"file": (os.path.basename(args.audio), f, "application/octet-stream")}
            r = requests.post(url, files=files, data=data, timeout=args.timeout)
    except requests.RequestException as e:
        print("ERROR: request failed: %s" % e, file=sys.stderr)
        sys.exit(1)

    if r.status_code != 200:
        # 服务端统一返回 {"detail": ...}
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        print("ERROR %d: %s" % (r.status_code, detail), file=sys.stderr)
        sys.exit(1)

    payload = r.json()
    sentences = payload.get("sentences", [])

    print("\n--- diarization result ---")
    for s in sentences:
        if "spk" in s and s.get("spk") is not None:
            print("[{:>6.2f}s-{:>6.2f}s] speaker {}: {}".format(
                s.get("start", 0), s.get("end", 0), s.get("spk"), s.get("text", "")))
        else:
            print(s.get("text", ""))

    nspk = len({s.get("spk") for s in sentences if s.get("spk") is not None})
    print("\nspeakers: %d, segments: %d" % (nspk, len(sentences)))

    # 完整 JSON 便于程序化处理
    print("\n--- raw json ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
