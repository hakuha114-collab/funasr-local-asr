# -*- encoding: utf-8 -*-
"""
多人语音说话人分离（Diarization）离线脚本 —— 用于解决"多人录音无法按声纹区分说话人"的痛点。

运行环境: conda（见仓库根目录 environment.yml，已锁定 funasr==1.3.30）
  conda env create -f environment.yml
  conda activate funasr
  python diarize.py --audio_in meeting.wav

相比 WebSocket 在线服务，这条 Python 路线对说话人分离支持最完整、最可靠：
  VAD + ASR(Paraformer) + 标点 + 说话人嵌入(CAM++) 一条管线，直接返回每句话的说话人 id。

模型: 首次运行自动从 ModelScope 下载 paraformer-zh / fsmn-vad / ct-punc / cam++。

注意（官方已知限制，效果受音频质量影响）:
  - 建议音频时长 > 40s，太短难以聚类的说话人
  - 16k 单声道、清晰、少噪声、说话人音色差异明显时效果最好
  - 说话人音色相近或重叠语音多时，区分准确率会下降

输出（--output_dir 时）:
  - diarize.json      结构化结果（带时间戳/说话人/文本）
  - diarize.txt       人类可读的逐句转录（带 [起-止]s speaker N: 文本）
  - diarize_debug.json 原始 model.generate 返回值，便于排查"文本为空"等问题

用法:
  python diarize.py --audio_in meeting.wav
  python diarize.py --audio_in meeting.wav --output_dir ./results
  python diarize.py --audio_in meeting.wav --max_speakers 3 --device cuda
  # 热词提升专有名词识别（解码阶段偏置）:
  python diarize.py --audio_in meeting.wav --hotword "科大讯飞 特斯拉 贝叶斯"
  python diarize.py --audio_in meeting.wav --hotword_file ./hotwords.txt
  # 数字默认按阿拉伯数字输出(itn 默认关, 保留 123456); 若想转成中文数字(一百二十三)则加 --itn:
  python diarize.py --audio_in meeting.wav --itn
"""
import argparse
import os
import json
from datetime import datetime

from funasr import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--audio_in", type=str, required=True, help="输入音频路径(wav/pcm/mp4)")
parser.add_argument("--output_dir", type=str, default=None, help="输出目录, 不填则仅打印")
parser.add_argument("--model", type=str, default="paraformer-zh", help="ASR 模型, 默认 paraformer-zh")
parser.add_argument("--max_speakers", type=int, default=0, help="最大说话人数, 0=自动")
parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
parser.add_argument("--hotword", type=str, default=None, help="热词(空格分隔), 提升专有名词识别, 如 '科大讯飞 特斯拉 贝叶斯'")
parser.add_argument("--hotword_file", type=str, default=None, help="热词文件, 每行一个词, 与 --hotword 合并生效")
parser.add_argument("--itn", dest="itn", action="store_true", default=False,
                    help="逆文本归一化: 把阿拉伯数字转成中文数字(如 123->一百二十三)。默认关, 保留 123456")
parser.add_argument("--no_itn", dest="itn", action="store_false",
                    help="(已废弃, 默认即为关闭) 保持模型原始阿拉伯数字格式")
args = parser.parse_args()

# 组装热词（命令行 + 文件）
hotword = args.hotword
if args.hotword_file and os.path.exists(args.hotword_file):
    with open(args.hotword_file, "r", encoding="utf-8") as f:
        file_words = [ln.strip() for ln in f if ln.strip()]
    hw = " ".join(file_words)
    hotword = (hotword + " " + hw).strip() if hotword else hw
    print("[热词] 已从文件加载 {} 个词".format(len(file_words)))

# 输出文件前缀: 音频名 + 日期(yyyymmdd), 例如 meeting_20260826
_audio_base = os.path.splitext(os.path.basename(args.audio_in))[0]
_date_tag = datetime.now().strftime("%Y%m%d")
out_prefix = "{}_{}".format(_audio_base, _date_tag)

if args.output_dir and not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def build_model(with_spk=True):
    kwargs = dict(model=args.model, vad_model="fsmn-vad", punc_model="ct-punc", device=args.device)
    if with_spk:
        kwargs["spk_model"] = "cam++"
    return AutoModel(**kwargs)


model = build_model(with_spk=True)

gen_kwargs = {"input": args.audio_in, "batch_size_s": 300, "spk_diarization": True, "itn": args.itn}
if args.max_speakers > 0:
    gen_kwargs["max_speakers"] = args.max_speakers
if hotword:
    gen_kwargs["hotword"] = hotword

res = model.generate(**gen_kwargs)

# 1) dump 完整原始结果，便于排查字段差异
if args.output_dir:
    with open(os.path.join(args.output_dir, "{}_debug.json".format(out_prefix)), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def seg_text(s):
    # 不同 funasr 版本文本字段名可能不同，全部兜底试一遍
    return s.get("sentence") or s.get("text") or s.get("raw_text") or ""


sentences = []
if res and "sentence_info" in res[0]:
    for s in res[0]["sentence_info"]:
        start = s.get("start", 0) / 1000.0
        end = s.get("end", 0) / 1000.0
        spk = s.get("spk")
        text = seg_text(s)
        print("[{:.1f}s-{:.1f}s] speaker {}: {}".format(start, end, spk, text))
        sentences.append({"start": start, "end": end, "spk": spk, "text": text})
elif res and "text" in res[0]:
    full = res[0]["text"]
    print(full)
    sentences.append({"text": full})
else:
    print("未获取到结果，请检查音频文件。")

# 2) 分段文本全空时，回退到整段文本 res[0]["text"]
if sentences and all(not s.get("text") for s in sentences) and res and res[0].get("text"):
    print("\n[回退] 分段文本为空，但检测到整段文本，已使用 res[0]['text']：")
    print(res[0]["text"])
    sentences = [{"text": res[0]["text"]}]

# 3) 仍为空：做一次纯 ASR（不带说话人）兜底，至少拿到转录文本
if sentences and all(not s.get("text") for s in sentences):
    print("\n[兜底] 说话人分离管线未产出文本，尝试纯 ASR 回退...")
    try:
        m2 = build_model(with_spk=False)
        r2 = m2.generate(input=args.audio_in, batch_size_s=300, itn=args.itn,
                        **({"hotword": hotword} if hotword else {}))
        if r2 and r2[0].get("text"):
            print(r2[0]["text"])
            sentences = [{"text": r2[0]["text"]}]
        else:
            print("纯 ASR 也未产出文本 —— 大概率是音频本身无有效语音/解码失败，请检查文件格式(建议 16k 单声道 wav)。")
    except Exception as e:
        print("纯 ASR 回退失败: {}".format(e))

if args.output_dir:
    json_path = os.path.join(args.output_dir, "{}.json".format(out_prefix))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(args.output_dir, "{}.txt".format(out_prefix))
    with open(txt_path, "w", encoding="utf-8") as f:
        if len(sentences) == 1 and "start" not in sentences[0]:
            f.write(sentences[0].get("text", ""))
        else:
            for s in sentences:
                f.write("[{:.1f}s-{:.1f}s] speaker {}: {}\n".format(
                    s.get("start", 0), s.get("end", 0), s.get("spk"), s.get("text", "")))
    debug_path = os.path.join(args.output_dir, "{}_debug.json".format(out_prefix))
    print("\n结果已写入:\n  {}\n  {}\n  {}".format(json_path, txt_path, debug_path))
