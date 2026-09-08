# -*- encoding: utf-8 -*-
"""
Local diarization HTTP service (no Docker, uses conda env "funasr").

This is the RELIABLE multi-speaker path. It wraps funasr's AutoModel with cam++
speaker diarization (spk_diarization=True) — the real clustering that the
WebSocket service cannot do (the WS service only does speaker VERIFICATION
against an enrolled speaker_db.json, so everyone comes back "unknown").

It also handles ALL audio formats (wav/mp3/m4a/flac/ogg/aac/wma/mp4) because
funasr reads audio via ffmpeg, unlike the WS service which treats bytes as
raw PCM (so only .wav works there).

Endpoints:
  GET  /health        -> {"status": "ok"}
  POST /diarize       multipart form:
                        file        (required, audio file)
                        itn         (0/1, default 0 -> keep Arabic digits 123456)
                        hotword     (space separated, optional)
                        max_speakers (int, 0 = auto)
                      returns {"wav_name":..., "text":...,
                               "sentences":[{"start","end","spk","text"}]}

Run:
  conda activate funasr
  python diarize_server.py --port 10098 --device cpu
"""
import argparse
import asyncio
import os
import tempfile
import time
from datetime import datetime

from fastapi.concurrency import run_in_threadpool

# --- Offline model loading: force modelscope to use the local cache only ---
# FunASR's download path does not forward local_files_only, and it pings the
# hub to check "is local model latest", which fails offline / behind a proxy.
# Patch snapshot_download so it never touches the network (no re-download).
import importlib as _il
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

from funasr import AutoModel

# 只允许一个 diarize 请求占用模型；其它请求排队，避免同步 model.generate 阻塞事件循环。
DIARIZE_LOCK = asyncio.Lock()

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", help="bind host")
parser.add_argument("--port", type=int, default=10098, help="HTTP port")
parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
args = parser.parse_args()

print("diarize_server: loading models (paraformer + fsmn-vad + ct-punc + cam++)...")
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
print("diarize_server: models loaded, serving on port %d" % args.port)


def seg_text(s):
    # 不同 funasr 版本文本字段名可能不同，全部兜底试一遍
    return s.get("sentence") or s.get("text") or s.get("raw_text") or ""


def _parse(res, wav_name):
    """把 model.generate 返回值规范化成 (full_text, sentences[])，含多层兜底。"""
    sentences = []
    full_text = ""

    if res and "sentence_info" in res[0]:
        for s in res[0]["sentence_info"]:
            start = s.get("start", 0) / 1000.0
            end = s.get("end", 0) / 1000.0
            spk = s.get("spk")
            text = seg_text(s)
            sentences.append({"start": round(start, 2), "end": round(end, 2),
                               "spk": spk, "text": text})
        full_text = "\n".join(
            "[{:.1f}s-{:.1f}s] speaker {}: {}".format(s["start"], s["end"], s["spk"], s["text"])
            for s in sentences
        )
    elif res and "text" in res[0]:
        full_text = res[0]["text"]
        sentences = [{"text": full_text}]

    # 分段文本全空但有整段文本 -> 回退整段
    if sentences and all(not s.get("text") for s in sentences) and res and res[0].get("text"):
        sentences = [{"text": res[0]["text"]}]
        full_text = res[0]["text"]

    return full_text, sentences


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FunASR Diarization Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    itn: int = Form(0),
    hotword: str = Form(""),
    max_speakers: int = Form(0),
):
    t0 = time.time()
    content = await file.read()
    size_kb = len(content) / 1024.0 if content else 0.0
    print("=" * 60, flush=True)
    print("[DIARIZE] received: file=%s size=%.1fKB itn=%d max_speakers=%d hotword=%r"
          % (file.filename, size_kb, itn, max_speakers, hotword), flush=True)

    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    tmp = None
    try:
        if not content:
            raise HTTPException(status_code=400, detail="empty file")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp = f.name

        # 直接把原始上传文件喂给 funasr（与命令行 diarize.py 一致，funasr 内部会自行
        # 处理 8k/16k、单/双声道；做过重采样反而触发 torchcodec 报错导致整请求失败）
        gen_kwargs = {
            "input": tmp,
            "batch_size_s": 300,
            "spk_diarization": True,
            "itn": bool(itn),
        }
        if max_speakers > 0:
            gen_kwargs["max_speakers"] = max_speakers
        if hotword and hotword.strip():
            gen_kwargs["hotword"] = hotword.strip()

        # 直接在事件循环里调用同步 model.generate（会阻塞事件循环）。
        # 实测在子线程/线程池里调用会导致进程崩溃，故保留单线程串行。
        print("[DIARIZE] processing: %s ... (ASR + cam++ clustering, please wait)"
              % file.filename, flush=True)
        async with DIARIZE_LOCK:
            res = model.generate(**gen_kwargs)
        print("[DIARIZE] model finished: %s in %.1fs" % (file.filename, time.time() - t0), flush=True)
        raw_text = (res[0].get("text", "") if res else "") or ""
        print("[DIARIZE_RAW] %s -> text(%d chars): %s" % (file.filename, len(raw_text), raw_text[:200]), flush=True)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[DIARIZE] ERROR: %s -> %s" % (file.filename, str(e)[:300]), flush=True)
        raise HTTPException(status_code=500, detail="model inference failed: %s" % str(e))
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

    full_text, sentences = _parse(res, file.filename)
    if not sentences:
        print("[DIARIZE] ERROR: %s -> no transcription result from model" % file.filename, flush=True)
        raise HTTPException(status_code=422, detail="no transcription result from model")

    nspk = len({s.get("spk") for s in sentences if s.get("spk") is not None})
    print("[DIARIZE] done: %s -> speakers=%d segments=%d total %.1fs"
          % (file.filename, nspk, len(sentences), time.time() - t0), flush=True)

    return {
        "wav_name": file.filename,
        "text": full_text,
        "sentences": sentences,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
