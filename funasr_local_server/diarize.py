# -*- encoding: utf-8 -*-
"""
Standalone offline speaker-diarization CLI (no server needed).

This is the RELIABLE multi-speaker path as a command-line tool. It wraps
funasr's AutoModel with cam++ speaker diarization (spk_diarization=True) —
the real clustering, unlike the WS service which only does speaker
VERIFICATION against an enrolled speaker_db.json.

Usage:
  python diarize.py audio.wav
  python diarize.py ./audios/ --out_dir ./results --format both
  python diarize.py talk.mp3 --max_speakers 3 --hotword "阿里巴巴 20"
  python diarize.py meeting.wav --device cuda --itn 1

Notes:
  - Models load from the LOCAL ModelScope cache (offline, no download).
  - wav/flac/ogg work out of the box; mp3/m4a/aac/wma/mp4 need ffmpeg installed
    (funasr decodes non-wav via torchcodec/ffmpeg). Convert to 16k wav if a
    format fails, e.g.: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav
"""
import argparse
import importlib as _il
import json
import os
import sys

# --- Offline model loading: force modelscope to use the local cache only ---
# FunASR's download path does not forward local_files_only, and it pings the
# hub to check "is local model latest", which fails offline / behind a proxy.
# Patch snapshot_download so it never touches the network (no re-download).
_here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.normpath(
    os.path.join(_here, "..", "models", "modelscope")))

_ms_sd = _il.import_module("modelscope.hub.snapshot_download")
_ms_sd_orig = _ms_sd.snapshot_download


def _smart_snapshot_download(model, revision=None, **kwargs):
    try:
        # Fast path: use the local cache only, never touch the network.
        return _ms_sd_orig(model, revision=revision, local_files_only=True, **kwargs)
    except Exception:
        # Model missing/corrupt locally -> one-time download into the cache,
        # then it stays offline on later runs.
        print("[modelscope] '%s' not in local cache; downloading once..." % model)
        return _ms_sd_orig(model, revision=revision, local_files_only=False, **kwargs)


_ms_sd.snapshot_download = _smart_snapshot_download
# ---------------------------------------------------------------------------

from funasr import AutoModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Offline cam++ speaker diarization CLI")
    p.add_argument("input", help="audio file or a directory of audio files")
    p.add_argument("--out_dir", default="diarize_out",
                   help="where to write results (default: ./diarize_out)")
    p.add_argument("--format", choices=["txt", "json", "both"], default="both",
                   help="output format (default: both)")
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--max_speakers", type=int, default=0,
                   help="speaker count cap; 0 = auto (default: 0)")
    p.add_argument("--hotword", default="",
                   help='space-separated hotwords, e.g. "阿里巴巴 20"')
    p.add_argument("--itn", type=int, default=0,
                   help="1 = inverse text normalization (digits->words), 0 = keep digits")
    p.add_argument("--batch_size_s", type=int, default=300, help="VAD batch size (s)")
    p.add_argument("--recursive", action="store_true",
                   help="recurse into subdirectories when input is a directory")
    p.add_argument("--ext", default="wav,flac,ogg,m4a,mp3,aac,wma,mp4",
                   help="comma-separated audio extensions to accept in directory mode")
    return p.parse_args()


def seg_text(s):
    # 不同 funasr 版本文本字段名可能不同，全部兜底试一遍
    return s.get("sentence") or s.get("text") or s.get("raw_text") or ""


def parse_result(res):
    """把 model.generate 返回值规范化成 (full_text, sentences[])，含多层兜底。"""
    sentences = []
    full_text = ""

    if res and "sentence_info" in res[0]:
        for s in res[0]["sentence_info"]:
            start = s.get("start", 0) / 1000.0
            end = s.get("end", 0) / 1000.0
            sentences.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "spk": s.get("spk"),
                "text": seg_text(s),
            })
        full_text = "\n".join(
            "[{:.1f}s-{:.1f}s] speaker {}: {}".format(
                x["start"], x["end"], x["spk"], x["text"])
            for x in sentences
        )
    elif res and "text" in res[0]:
        full_text = res[0]["text"]
        sentences = [{"text": full_text}]

    # 分段文本全空但有整段文本 -> 回退整段
    if sentences and all(not x.get("text") for x in sentences) and res and res[0].get("text"):
        sentences = [{"text": res[0]["text"]}]
        full_text = res[0]["text"]

    return full_text, sentences


def collect_inputs(path, exts, recursive):
    """返回待处理的音频文件绝对路径列表（保持顺序）。"""
    exts = {("." + e.lower().lstrip(".")) for e in exts.split(",") if e.strip()}
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    found = []
    if recursive:
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    found.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in exts:
                found.append(fp)
    return sorted(found)


def diarize_one(model, args, audio_path):
    gen_kwargs = {
        "input": audio_path,
        "batch_size_s": args.batch_size_s,
        "spk_diarization": True,
        "itn": bool(args.itn),
    }
    if args.max_speakers > 0:
        gen_kwargs["max_speakers"] = args.max_speakers
    if args.hotword and args.hotword.strip():
        gen_kwargs["hotword"] = args.hotword.strip()
    res = model.generate(**gen_kwargs)
    return parse_result(res)


def main():
    args = parse_args()
    files = collect_inputs(args.input, args.ext, args.recursive)
    if not files:
        print("ERROR: no audio files found for input: %s" % args.input, file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    print("diarize: loading models (seaco-paraformer + fsmn-vad + ct-punc + cam++)...")
    print("diarize: MODELSCOPE_CACHE = %s" % os.environ.get("MODELSCOPE_CACHE"))
    model = AutoModel(
        model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        model_revision="v2.0.4",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_model_revision="v2.0.4",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        punc_model_revision="master",
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        spk_model_revision="master",
        device=args.device,
        disable_pbar=True,
        disable_log=True,
        disable_update=True,
    )
    print("diarize: models loaded. processing %d file(s)...\n" % len(files))

    ok, fail = 0, 0
    for audio_path in files:
        name = os.path.basename(audio_path)
        stem = os.path.splitext(name)[0]
        print("=" * 70)
        print("[audio] %s" % audio_path)
        try:
            full_text, sentences = diarize_one(model, args, audio_path)
        except Exception as e:  # noqa: BLE001
            fail += 1
            msg = str(e)
            hint = ""
            if any(k in msg.lower() for k in ("ffmpeg", "torchcodec", "decode", "format", "codec")):
                hint = ("\n  hint: this audio format likely needs ffmpeg. "
                        "install ffmpeg or convert to 16k wav first:\n"
                        "    ffmpeg -i \"%s\" -ar 16000 -ac 1 \"%s.wav\""
                        % (audio_path, stem))
            print("  FAILED: %s%s" % (msg[:300], hint))
            continue

        if not sentences:
            fail += 1
            print("  FAILED: no transcription result from model")
            continue

        nspk = len({s.get("spk") for s in sentences if s.get("spk") is not None})
        print(full_text)
        print("  -> speakers detected: %d, segments: %d" % (nspk, len(sentences)))

        if args.format in ("txt", "both"):
            txt_path = os.path.join(args.out_dir, stem + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text + "\n")
        if args.format in ("json", "both"):
            json_path = os.path.join(args.out_dir, stem + ".json")
            payload = {
                "audio": audio_path,
                "num_speakers": nspk,
                "num_segments": len(sentences),
                "sentences": sentences,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print("  -> wrote: %s" % args.out_dir)
        ok += 1

    print("\n" + "=" * 70)
    print("done: %d succeeded, %d failed. results in %s" % (ok, fail, args.out_dir))
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
