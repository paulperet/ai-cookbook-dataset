이 노트북은 OpenAI의  **gpt-oss (open‑weight)** 모델을 **한국 뉴스 문체 + 최신 대화체**로 세밀 튜닝하는 방법을
한국어/영어 **이중 언어**로 제공합니다.  
This notebook shows how to fine‑tune OpenAI's **gpt-oss (open‑weight)** models for **Korean news style + modern chat tone**, in **Korean & English**.

---

### MXFP4 workflow clarifications · MXFP4 워크플로 정리

**EN:**  
- Training or fine-tuning **directly in MXFP4 is not supported** by public frameworks today.  
- Recommended path: train in **BF16** (or **QLoRA 4‑bit nf4**) → **merge LoRA** → **post‑training quantize to MXFP4** → `save_pretrained()` for deployment.  
- If you need an MXFP4 artifact, you must **re‑quantize from BF16** after merging adapters. (Export utilities are evolving; if your toolchain already supports MXFP4 serialization, that’s ideal.)

**KR:**  
- 현재 공개 프레임워크에서는 **MXFP4로 직접 학습/파인튜닝**이 지원되지 않습니다.  
- 권장 경로: **BF16**(또는 **QLoRA 4‑bit nf4**)로 학습 → **LoRA 병합** → **사후(MXFP4) 양자화** → 배포용으로 `save_pretrained()` 저장.  
- MXFP4 아티팩트가 필요하면, 어댑터 병합 후 **BF16 → MXFP4 재양자화**가 필요합니다. (직렬화 유틸은 진화 중이며, 툴체인에서 MXFP4 저장을 지원하면 가장 좋습니다.)

---

### LoRA targets (MoE) · LoRA 타깃(MoE 포함)

**EN:**  
- Minimal config (fast, low VRAM): target attention only, e.g. `["q_proj","v_proj"]`.  
- MoE‑aware config (better domain adaptation, more VRAM/time): include **expert projection layers** in addition to attention.  

```python
from peft import LoraConfig

TARGET_MODULES = ["q_proj", "v_proj"]  # baseline
MOE_TARGET_PARAMETERS = [
    # example expert layers; adjust indices to your model depth
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",              # cover all linear layers
    target_parameters=MOE_TARGET_PARAMETERS,  # add expert projections
    bias="none", task_type="CAUSAL_LM",
)
```

- Start with attention‑only; if KR domain fit is insufficient, enable MoE targets and re‑eval.

**KR:**  
- 최소 구성(빠르고 VRAM 절약): `["q_proj","v_proj"]` 등 **어텐션만** 적용.  
- **MoE 인지 구성**(도메인 적합성↑, 자원 소모↑): 어텐션에 **전문가(Expert) 투영 레이어**를 추가로 포함.  
- 먼저 어텐션만으로 시도한 뒤, 한국어 도메인 적합성이 부족하면 MoE 타깃을 켜고 재평가하세요.

## Contents · 목차
0) Goals & Scope · 목표 & 범위  
1) Environment check · 환경 점검  
2) 설정값 · Config  
3) 패키지 설치 · Install Deps  
4) 데이터 소싱(한국형) · KR‑Context Data Sourcing  
5) 샘플 데이터 생성 · Create Sample Data  
6) 전처리(PIPA) & 스타일 라벨 · PII Scrubbing & Style Tags  
7) 데이터 로딩/포맷팅 · Load & Format  
8) 모델/토크나이저 로드 · Load Model & Tokenizer  
9) Fine‑Tuning (LoRA/QLoRA) · 세밀 튜닝  
   9a) Data curation & splits  
   9b) Hyperparameters (r/alpha/dropout)  
   9c) Merge adapters (BF16)  
   9d) Save merged BF16 (`save_pretrained`)  
   9e) Export & Quantize (BF16 → MXFP4) · 내보내기 & 양자화  
10) 평가(뉴스/대화) · Evaluation (News/Chat)  
11) Inference Prompt Templates · 추론 프롬프트 템플릿  
12) 최신성 유지 · Freshness Strategy  
13) 안전/컴플라이언스 · Safety & Compliance  
14) 문제해결 & 다음 단계 · Troubleshooting & Next Steps


### ⚙️ Training vs Quantization — What’s supported
- **Do:** Train with BF16/FP16 or QLoRA; export merged weights.
- **Then:** Quantize to **MXFP4** for inference using provided conversion scripts/utilities.
- **Don’t:** Attempt to run an end‑to‑end “train in MXFP4” pipeline — not supported today.

> **PII & Compliance Reminder:** For KR data, follow your enterprise policy (mask RRN/phone/account IDs, remove emails) **before** training & logging. Keep train/val/test splits stratified by source and style tags.

### 🧪 MoE adapters (optional)
You can target MoE layers with adapters, but treat this as **advanced/experimental**. Start with attention projections first and validate KR benchmarks before expanding scope.

> **Note:** Keep `transformers`, `peft`, `accelerate`, and `trl` at versions known to support BF16/4‑bit LoRA.  
If you pin `safetensors`, remember that **native MXFP4 serialization is not yet standardized**; loaders may upcast internally.

### 🔎 Support Matrix — At a glance
- **Fine‑tuning precision:** BF16/FP16 ✅ · QLoRA 4‑bit ✅ · **MXFP4 FT ❌**
- **Quantization target:** MXFP4 ✅ (post‑training)
- **API FT (hosted) for OSS models:** ❌
- **Open‑source FT (Transformers/TRL/PEFT):** ✅
- **LoRA targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj` ✅; MoE expert adapters **experimental** ⚠️

---

## 0) Goals & Scope · 목표 & 범위
- **KR**: 한국어 일반 뉴스 + 일상/상담 대화체에 최적화. `style=news_headline|news_lead|news_body|kakao_casual|kakao_formal` 제어.
- **EN**: Optimize for Korean news writing and modern chat tone; control output via style tags above.
- **Stack**: `transformers`, `trl(SFTTrainer)`, `peft(LoRA/QLoRA)`, `datasets`.
- **Hardware**: Single/few GPUs (BF16 preferred). CPU/Mac for lightweight tests.

## 1) Environment check · 환경 점검


```python
import os, sys, platform
print("Python:", sys.version)
print("OS/Platform:", platform.platform())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

try:
    import torch
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch not installed or GPU not detected:", e)
```

    Python: 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]
    OS/Platform: Linux-6.8.0-60-generic-x86_64-with-glibc2.35
    CUDA_VISIBLE_DEVICES: 
    Torch: 2.7.1+cu126 CUDA: True
    GPU: NVIDIA H100 80GB HBM3


## 2) 설정값 · Config


```python
from pathlib import Path
import os

# === Model & Training Params ===
BASE_URL = "http://localhost:8000/v1"     # vLLM OpenAI-compatible endpoint
API_KEY  = "dummy-key"                     # vLLM ignores; SDK requires a value
MODEL    = "openai/gpt-oss-120b"           # must match the model vLLM loaded
OUTPUT_DIR = "ft-oss-kr-news-chat-bilingual"

# Data mix (news : chat)
MIX_NEWS = 0.6
MIX_CHAT = 0.4

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]  # adjust per model

# Training
EPOCHS = 1
PER_DEVICE_BS = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
BF16 = True
LOG_STEPS = 20
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2

print("Config ready.")
```

    Config ready.


## 3) 패키지 설치 · Install Deps


```python
# %pip install --upgrade pip
# %pip install transformers accelerate datasets peft trl bitsandbytes sentencepiece
# (optional) serving/runtimes
# %pip install vllm
# %pip install llama-cpp-python

import importlib, pip

for dep in ["transformers","accelerate","datasets","peft","trl",
            "bitsandbytes","sentencepiece","vllm","llama_cpp"]:
    try:
        print(f"{dep}: {importlib.import_module(dep).__version__}")
    except Exception:
        print(f"{dep}: not installed")

print(f"pip: {pip.__version__}")

print("Install cells are commented. Un-comment in your environment.")
```

    transformers: 4.55.3
    accelerate: 1.10.0
    datasets: 4.0.0
    peft: not installed
    trl: 0.21.0
    bitsandbytes: not installed
    sentencepiece: 0.2.1
    vllm: 0.10.1
    llama_cpp: 0.3.16
    pip: 25.2
    Install cells are commented. Un-comment in your environment.


## 4) 데이터 소싱(한국형) · KR‑Context Data Sourcing

**KR**  
- 공개 벤치마크(주제 분류/요약/QA) + **허용된 뉴스 API의 메타데이터(제목/요약/섹션)** 중심으로 스타일 보정.
- 기사 **원문 대량 재학습은 저작권/약관 이슈** → 메타데이터·공개 코퍼스 위주.
- 대화체는 합법 공개 코퍼스(반말/존댓말/이모티콘/축약어 라벨 포함) 우선.
- PIPA: 주민번호/연락처/이메일/계좌 등 개인정보는 **훈련 전/로그 전** 스크러빙.

**EN**  
- Prefer public KR benchmarks (topic classification / summarization / QA) and **allowed news API metadata** for style calibration.
- Avoid mass training on news full texts due to license/ToS constraints; use metadata + open corpora.
- For chat, use lawful open corpora with tone/emoji/informal‑formal annotations.
- Scrub PII (phone, RRNs, emails, accounts) before training/logging.

## 5) 샘플 데이터 생성 · Create Sample Data


```python
import json, pathlib
pathlib.Path("data").mkdir(exist_ok=True)

news_samples = [
  {"style":"news_lead","topic":"경제","title":"반도체 수출 호조… 7월 수출액 20% 증가","summary":"수출 개선세가 이어지며 경기 회복 기대가 커졌다."},
  {"style":"news_headline","topic":"정치","title":"국회, 데이터 산업 육성법 본회의 통과","summary":"데이터 활용 촉진과 개인정보 보호를 강화하는 내용."},
  {
    "style": "news_lead",
    "topic": "경제",
    "title": "카카오페이 보안 점검… 고객문의: help+vip@corp.co.kr",
    "summary": "고객센터 010-1234-5678로 문의 폭주. 계좌 110-123-456789 관련 결제 오류 논란."
  },
  {
    "style": "news_headline",
    "topic": "사회",
    "title": "개인정보 유출 의혹… 주민번호 901010-1234567 유통 주장",
    "summary": "서울특별시 강남구 테헤란로 123에서 자료 확보… 담당자 john.doe+news@example.com"
  }
]

chat_samples = [
  {"style":"kakao_casual","dialog":["주말에 비 온대?","응 일요일에 꽤 온다더라 ☔","헐 우산 챙겨야겠다"]},
  {"style":"kakao_formal","dialog":["안녕하세요. 배송 일정 확인 부탁드립니다.","내일 중 도착 예정입니다.","안내 감사합니다."]},
  {
    "style": "kakao_formal",
    "dialog": [
      "배송 확인 부탁드립니다. 주문번호 ORD-2025-0001 입니다.",
      "연락처는 010-2222-3333 입니다. (유니코드 하이픈)",
      "주민등록번호는 제공할 수 없습니다."
    ]
  }
]

with open("data/news.jsonl","w",encoding="utf-8") as f:
    for ex in news_samples: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
with open("data/chat.jsonl","w",encoding="utf-8") as f:
    for ex in chat_samples: f.write(json.dumps(ex, ensure_ascii=False)+"\n")

print("Created: data/news.jsonl, data/chat.jsonl")
```

    Created: data/news.jsonl, data/chat.jsonl


## 6) 전처리(PIPA) & 스타일 라벨 · PII Scrubbing & Style Tags


```python
# Step 6 — PII scrubbing + style tags (no Harmony here)
import json, re, unicodedata
from pathlib import Path

# --- Normalization helpers ---
HYPHENS = dict.fromkeys(map(ord, "‐-‒–—―﹘﹣－"), ord("-"))  # map unicode hyphens → ASCII
def normalize(s: str) -> str:
    if not isinstance(s, str): return s
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(HYPHENS)
    return s

# --- PII patterns (illustrative; tune for production) ---
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# KR mobile numbers with spaces/hyphens: 010-1234-5678, 010 1234 5678, etc.
RE_PHONE = re.compile(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b")
# Korean RRN (주민등록번호) basic pattern
RE_RRN = re.compile(r"\b\d{6}-\d{7}\b")
# Bank-ish account numbers: strictly digits in groups (avoid codes with letters)
RE_ACCOUNT = re.compile(r"\b\d{2,3}-\d{2,4}-\d{3,6}\b")
# Very simple postal address cue (city names) – conservative, just redact the token (optional)
RE_CITY = re.compile(r"(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|경기도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도|제주특별자치도)")

# Allowlist: things that look like PII but aren’t (e.g., bill/order codes w/ letters)
def looks_like_code(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))  # if letters present, treat as code, not account/phone

# Order of application matters (longest/most specific first sometimes helps)
SCRUBBERS = [
    ("[RRN]", RE_RRN),
    ("[EMAIL]", RE_EMAIL),
    ("[PHONE]", RE_PHONE),
    ("[ACCOUNT]", RE_ACCOUNT),
    ("[CITY]", RE_CITY),  # optional; comment out if you don't want to redact city tokens
]

def scrub_text(text: str) -> tuple[str, dict]:
    """Return (scrubbed_text, hits_dict). Avoid false positives with basic allowlisting."""
    if not isinstance(text, str) or not text:
        return text, {}
    orig = text
    text = normalize(text)
    hits = {}

    # Guard account-like and phone-like strings that contain letters (likely codes)
   