# Fine-Tuning OpenAI's gpt-oss Models for Korean News and Chat Style

This guide demonstrates how to fine-tune OpenAI's **gpt-oss (open-weight)** models to generate text in **Korean news style and modern chat tone**. The process is presented bilingually (Korean/English) and covers the complete workflow from data preparation to model quantization.

## Key Workflow Clarifications

**Training and Quantization:**
- **Direct training in MXFP4 is not supported** by public frameworks.
- **Recommended Path:**
    1. Train using **BF16** or **QLoRA 4-bit nf4**.
    2. Merge the LoRA adapter.
    3. Perform **post-training quantization to MXFP4**.
    4. Save the model for deployment with `save_pretrained()`.
- If you require an MXFP4 model, you must **re-quantize from the merged BF16 weights**. Export utilities are evolving, so check if your toolchain supports MXFP4 serialization.

**LoRA Targets (Including MoE):**
- **Minimal Configuration (Fast, Low VRAM):** Target only attention layers (e.g., `["q_proj","v_proj"]`).
- **MoE-aware Configuration (Better Domain Adaptation, More VRAM/Time):** Include expert projection layers in addition to attention.
- Start with attention-only targeting. If Korean domain adaptation is insufficient, enable MoE targets and re-evaluate.

## Contents
0. Goals & Scope
1. Environment Check
2. Configuration
3. Install Dependencies
4. Korean-Context Data Sourcing
5. Create Sample Data
6. PII Scrubbing & Style Tagging
7. Load & Format Data
8. Load Model & Tokenizer
9. Fine-Tuning (LoRA/QLoRA)
10. Evaluation (News/Chat)
11. Inference Prompt Templates
12. Freshness Strategy
13. Safety & Compliance
14. Troubleshooting & Next Steps

---

## 0. Goals & Scope
- **Korean (KR):** Optimize for general Korean news and daily/counseling chat tone. Control output via style tags: `news_headline`, `news_lead`, `news_body`, `kakao_casual`, `kakao_formal`.
- **English (EN):** Optimize for Korean news writing and modern chat tone; control output via the style tags above.
- **Stack:** `transformers`, `trl(SFTTrainer)`, `peft(LoRA/QLoRA)`, `datasets`.
- **Hardware:** Single or few GPUs (BF16 preferred). Use CPU/Mac for lightweight tests.

---

## 1. Environment Check

First, verify your Python environment and GPU availability.

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

**Expected Output:**
```
Python: 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]
OS/Platform: Linux-6.8.0-60-generic-x86_64-with-glibc2.35
CUDA_VISIBLE_DEVICES:
Torch: 2.7.1+cu126 CUDA: True
GPU: NVIDIA H100 80GB HBM3
```

---

## 2. Configuration

Define the core parameters for your fine-tuning run, including model paths, data mix, and LoRA settings.

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

---

## 3. Install Dependencies

Install the required Python packages. The following cell checks for their presence; uncomment the `pip install` lines in your environment if needed.

```python
import importlib, pip

# Check for installed dependencies
for dep in ["transformers","accelerate","datasets","peft","trl",
            "bitsandbytes","sentencepiece","vllm","llama_cpp"]:
    try:
        print(f"{dep}: {importlib.import_module(dep).__version__}")
    except Exception:
        print(f"{dep}: not installed")

print(f"pip: {pip.__version__}")

print("Install cells are commented. Un-comment in your environment.")
```

**Example Output:**
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
```

---

## 4. Korean-Context Data Sourcing

**Guidelines:**
- **Korean (KR):** Focus on public benchmarks (topic classification, summarization, QA) and **metadata from permitted news APIs** (titles, summaries, sections) for style calibration. Avoid mass retraining on full news articles due to copyright/license issues. For chat style, use lawful open corpora with annotations for tone, emojis, and formality. Scrub PII (Resident Registration Numbers, phone numbers, emails, accounts) **before training and logging**.
- **English (EN):** Prefer public Korean benchmarks and allowed news API metadata. Avoid training on full news texts due to license constraints. Use open chat corpora with tone/emoji/informal-formal annotations. Scrub PII before training/logging.

---

## 5. Create Sample Data

Create sample datasets for news and chat styles to illustrate the data format.

```python
import json, pathlib

# Create data directory
pathlib.Path("data").mkdir(exist_ok=True)

# Sample news data with style tags
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

# Sample chat data with style tags
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

# Write samples to JSONL files
with open("data/news.jsonl","w",encoding="utf-8") as f:
    for ex in news_samples: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
with open("data/chat.jsonl","w",encoding="utf-8") as f:
    for ex in chat_samples: f.write(json.dumps(ex, ensure_ascii=False)+"\n")

print("Created: data/news.jsonl, data/chat.jsonl")
```

---

## 6. PII Scrubbing & Style Tags

Before training, you must scrub Personally Identifiable Information (PII) from your data. This step is crucial for compliance and safety.

```python
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
    # Implementation continues...
```

*Note: The full scrubbing function implementation is truncated for brevity. Ensure you complete the logic to replace PII matches with placeholders like `[EMAIL]` or `[PHONE]`.*

---

## Next Steps

The subsequent steps (7-14) involve loading and formatting the scrubbed data, loading the model and tokenizer, configuring and running the fine-tuning process with LoRA/QLoRA, evaluating the model, and finally quantizing it to MXFP4 for efficient inference.

Remember to:
- Keep `transformers`, `peft`, `accelerate`, and `trl` at versions known to support BF16/4-bit LoRA.
- Follow your enterprise PII and compliance policy for Korean data.
- Treat MoE layer targeting with adapters as advanced/experimental; start with attention projections first.

**Support Matrix at a Glance:**
- **Fine-tuning precision:** BF16/FP16 ✅ · QLoRA 4-bit ✅ · **MXFP4 FT ❌**
- **Quantization target:** MXFP4 ✅ (post-training)
- **API FT (hosted) for OSS models:** ❌
- **Open-source FT (Transformers/TRL/PEFT):** ✅
- **LoRA targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj` ✅; MoE expert adapters **experimental** ⚠️