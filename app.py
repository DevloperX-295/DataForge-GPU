import asyncio
import csv
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =====================================================================
# 1. GPU ENGINE & MODEL MANAGER
# =====================================================================

class GPUEngine:
    def __init__(self):
        self.device = "cpu"
        self.device_name = "CPU (Multi-Core Fallback)"
        self.vram_total_gb = 0.0
        self.vram_allocated_gb = 0.0
        self.model = None
        self.tokenizer = None
        self.loaded_model_id = None
        self._detect_hardware()

    def _detect_hardware(self):
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                self.vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
                self.vram_allocated_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                self.device_name = "Apple Silicon GPU (MPS)"

    def get_status(self) -> Dict[str, Any]:
        self._detect_hardware()
        return {
            "torch_installed": TORCH_AVAILABLE,
            "device": self.device,
            "device_name": self.device_name,
            "vram_total_gb": self.vram_total_gb,
            "vram_allocated_gb": self.vram_allocated_gb,
            "loaded_model": self.loaded_model_id or "Turbo Heuristic Engine",
        }

    def load_llm(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch & Transformers are not installed.")
        if self.loaded_model_id == model_id and self.model is not None:
            return

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "mps":
            self.model.to("mps")
        elif self.device == "cpu":
            self.model.to("cpu")

        self.model.eval()
        self.loaded_model_id = model_id

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        if not self.model or not self.tokenizer:
            return []
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i, out in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            generated_tokens = out[input_len:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            results.append(text)
        return results


gpu_engine = GPUEngine()


# =====================================================================
# 2. 4-PASS DATA DEBUGGING ENGINE
# =====================================================================

class DataDebugger:
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    API_KEY_RE = re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35})\b")
    SSN_CC_RE = re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b")
    MULTI_SPACE_RE = re.compile(r"[ \t]+")
    MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    PAGE_NUM_RE = re.compile(r"(?m)^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|\d+)\s*$")

    @classmethod
    def pass1_clean_noise(cls, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = cls.HTML_TAG_RE.sub(" ", text)
        text = cls.PAGE_NUM_RE.sub("", text)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
        text = cls.MULTI_SPACE_RE.sub(" ", text)
        text = cls.MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()

    @classmethod
    def pass2_scrub_pii(cls, text: str) -> Tuple[str, int]:
        redactions = 0
        for pattern, replacement in [
            (cls.API_KEY_RE, "[API_KEY]"),
            (cls.EMAIL_RE, "[EMAIL]"),
            (cls.SSN_CC_RE, "[REDACTED_ID]"),
            (cls.PHONE_RE, "[PHONE]"),
            (cls.IPV4_RE, "[IP_ADDRESS]"),
        ]:
            text, count = pattern.subn(replacement, text)
            redactions += count
        return text, redactions

    @staticmethod
    def _shingles(text: str, k: int = 5) -> Set[int]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < k:
            return {hash(" ".join(words))}
        return {hash(" ".join(words[i : i + k])) for i in range(len(words) - k + 1)}

    @classmethod
    def pass3_deduplicate(cls, chunks: List[str], similarity_threshold: float = 0.82) -> Tuple[List[str], int]:
        seen_hashes: Set[str] = set()
        kept_chunks: List[str] = []
        kept_shingles: List[Set[int]] = []
        removed = 0

        for chunk in chunks:
            norm = re.sub(r"\s+", " ", chunk.lower().strip())
            exact_h = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if exact_h in seen_hashes:
                removed += 1
                continue

            sh = cls._shingles(norm)
            is_near_dup = False
            for existing_sh in kept_shingles[-250:]:
                intersection = len(sh & existing_sh)
                union = len(sh | existing_sh)
                if union > 0 and (intersection / union) >= similarity_threshold:
                    is_near_dup = True
                    break

            if is_near_dup:
                removed += 1
                continue

            seen_hashes.add(exact_h)
            kept_chunks.append(chunk)
            kept_shingles.append(sh)

        return kept_chunks, removed

    @classmethod
    def pass4_quality_filter(cls, chunk: str, min_words: int = 20) -> Tuple[bool, float]:
        words = chunk.split()
        if len(words) < min_words:
            return False, 0.0

        alpha_chars = sum(c.isalpha() for c in chunk)
        total_chars = max(len(chunk), 1)
        alpha_ratio = alpha_chars / total_chars

        if alpha_ratio < 0.55:
            return False, round(alpha_ratio, 2)

        avg_word_len = sum(len(w) for w in words) / len(words)
        has_sentence_punct = any(p in chunk for p in (".", "!", "?"))
        score = min(
            1.0,
            (alpha_ratio * 0.7)
            + (0.2 if has_sentence_punct else 0.0)
            + (0.1 if 3.5 <= avg_word_len <= 9.0 else 0.0),
        )
        return score >= 0.55, round(score, 2)


# =====================================================================
# 3. FILE & WEB LINK INGESTION ENGINE
# =====================================================================

class IngestionEngine:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
    }

    @staticmethod
    def chunk_text(text: str, target_words: int = 220) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current_words: List[str] = []

        for para in paragraphs:
            p_words = para.split()
            if len(current_words) + len(p_words) <= target_words:
                current_words.extend(p_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                if len(p_words) > target_words:
                    for i in range(0, len(p_words), target_words):
                        chunks.append(" ".join(p_words[i : i + target_words]))
                    current_words = []
                else:
                    current_words = p_words

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    @classmethod
    def extract_from_bytes(cls, filename: str, content: bytes) -> str:
        ext = os.path.splitext(filename.lower())[1]

        if ext == ".pdf":
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(stream=content, filetype="pdf")
                return "\n\n".join(page.get_text("text") for page in doc)
            elif PYPDF_AVAILABLE:
                reader = PdfReader(io.BytesIO(content))
                return "\n\n".join((page.extract_text() or "") for page in reader.pages)
            else:
                raise RuntimeError("Install 'pymupdf' or 'pypdf' to parse PDF files.")

        elif ext == ".csv":
            decoded = content.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(decoded))
            rows = [" | ".join(col.strip() for col in row if col.strip()) for row in reader]
            return "\n\n".join(r for r in rows if r)

        elif ext in (".json", ".jsonl"):
            decoded = content.decode("utf-8", errors="replace")
            blocks = []
            if ext == ".jsonl":
                for line in decoded.splitlines():
                    if line.strip():
                        blocks.append(line.strip())
            else:
                try:
                    data = json.loads(decoded)
                    if isinstance(data, list):
                        blocks = [json.dumps(item, ensure_ascii=False) for item in data]
                    else:
                        blocks = [json.dumps(data, ensure_ascii=False)]
                except Exception:
                    blocks = [decoded]
            return "\n\n".join(blocks)

        else:
            return content.decode("utf-8", errors="replace")

    @classmethod
    def fetch_url(cls, url: str) -> Tuple[str, str]:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        resp = requests.get(url, headers=cls.HEADERS, timeout=15)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()

        if url.endswith(".xml") or "xml" in content_type:
            try:
                root = ET.fromstring(resp.content)
                sub_urls = [elem.text.strip() for elem in root.iter() if elem.tag.endswith("loc") and elem.text]
                if sub_urls:
                    collected = []
                    for sub in sub_urls[:25]:
                        try:
                            _, sub_txt = cls.fetch_single_html(sub)
                            if sub_txt:
                                collected.append(sub_txt)
                        except Exception:
                            continue
                    return url, "\n\n".join(collected)
            except Exception:
                pass

        return cls.fetch_single_html(url, pre_fetched_html=resp.text)

    @classmethod
    def fetch_single_html(cls, url: str, pre_fetched_html: Optional[str] = None) -> Tuple[str, str]:
        html = pre_fetched_html
        if html is None:
            resp = requests.get(url, headers=cls.HEADERS, timeout=15)
            resp.raise_for_status()
            html = resp.text

        if TRAFILATURA_AVAILABLE:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
            if extracted and len(extracted.split()) >= 20:
                return url, extracted

        soup = BeautifulSoup(html, "html.parser")
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form"]):
            junk.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all(["p", "article", "section", "li", "h1", "h2", "h3"])
        ]
        clean_text = "\n\n".join(p for p in paragraphs if len(p.split()) >= 6)
        return url, clean_text


# =====================================================================
# 4. AI TRAINING FORMAT SYNTHESIZER
# =====================================================================

class DatasetSynthesizer:
    SYSTEM_PROMPT = "You are a helpful, accurate, and domain-expert AI assistant."

    @classmethod
    def extract_topic_and_question(cls, chunk: str) -> Tuple[str, str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        first_sent = sentences[0] if sentences else chunk[:120]
        words = re.findall(r"[A-Za-z0-9-]+", first_sent)
        topic = " ".join(words[:8]) if words else "the provided subject"
        question = f"Explain the key concepts and details regarding {topic}."
        return topic, question

    @classmethod
    def synthesize_record(
        cls,
        raw_chunk: str,
        clean_chunk: str,
        source: str,
        quality_score: float,
        schema: str,
        llm_generated_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        _, auto_question = cls.extract_topic_and_question(clean_chunk)
        instruction = llm_generated_instruction or auto_question
        token_est = int(len(clean_chunk.split()) * 1.33)

        if schema == "alpaca":
            formatted = {
                "instruction": instruction,
                "input": "",
                "output": clean_chunk,
            }
        elif schema == "sharegpt":
            formatted = {
                "conversations": [
                    {"from": "system", "value": cls.SYSTEM_PROMPT},
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": clean_chunk},
                ]
            }
        elif schema == "chatml":
            formatted = {
                "messages": [
                    {"role": "system", "content": cls.SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": clean_chunk},
                ]
            }
        elif schema == "dpo":
            short_rejected = clean_chunk.split(".")[0] + "." if "." in clean_chunk else "Information unavailable."
            formatted = {
                "prompt": instruction,
                "chosen": clean_chunk,
                "rejected": f"Incomplete summary: {short_rejected}",
            }
        else:
            formatted = {
                "text": clean_chunk,
                "meta": {"source": source, "quality_score": quality_score, "tokens": token_est},
            }

        return {
            "id": str(uuid.uuid4())[:8],
            "source": source,
            "raw_text": raw_chunk,
            "clean_text": clean_chunk,
            "quality_score": quality_score,
            "tokens": token_est,
            "formatted": formatted,
        }


# =====================================================================
# 5. IN-MEMORY DATASET STORE & FASTAPI ROUTES
# =====================================================================

@dataclass
class DatasetSession:
    records: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(
        default_factory=lambda: {
            "sources_processed": 0,
            "raw_chunks": 0,
            "debugged_chunks": 0,
            "duplicates_removed": 0,
            "low_quality_removed": 0,
            "pii_redactions": 0,
            "total_tokens": 0,
            "last_duration_sec": 0.0,
        }
    )


SESSION = DatasetSession()
app = FastAPI(title="DataForge GPU - AI Training Data Debugger")


@app.get("/api/status")
async def api_status():
    return {
        "gpu": gpu_engine.get_status(),
        "stats": SESSION.stats,
        "total_records": len(SESSION.records),
    }


@app.post("/api/load-model")
async def api_load_model(model_id: str = Form("Qwen/Qwen2.5-1.5B-Instruct")):
    try:
        gpu_engine.load_llm(model_id)
        return {"status": "ok", "gpu": gpu_engine.get_status()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/clear")
async def api_clear():
    SESSION.records.clear()
    for k in SESSION.stats:
        SESSION.stats[k] = 0 if isinstance(SESSION.stats[k], int) else 0.0
    return {"status": "cleared"}


@app.post("/api/process")
async def api_process(
    files: List[UploadFile] = File(default=[]),
    urls: str = Form(default=""),
    raw_text: str = Form(default=""),
    schema: str = Form(default="alpaca"),
    chunk_words: int = Form(default=220),
    min_words: int = Form(default=20),
    enable_pii: bool = Form(default=True),
    enable_dedup: bool = Form(default=True),
    use_gpu_llm: bool = Form(default=False),
):
    start_time = time.time()
    extracted_sources: List[Tuple[str, str]] = []
    errors: List[str] = []

    for upload in files:
        if not upload.filename:
            continue
        try:
            content = await upload.read()
            text = IngestionEngine.extract_from_bytes(upload.filename, content)
            if text.strip():
                extracted_sources.append((upload.filename, text))
        except Exception as e:
            errors.append(f"File {upload.filename}: {str(e)}")

    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if url_list:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(12, len(url_list))) as pool:
            tasks = [loop.run_in_executor(pool, IngestionEngine.fetch_url, u) for u in url_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    errors.append(f"URL {url_list[idx]}: {str(res)}")
                else:
                    src_url, txt = res
                    if txt.strip():
                        extracted_sources.append((src_url, txt))

    if raw_text.strip():
        extracted_sources.append(("Direct_Text_Input", raw_text))

    candidate_items: List[Tuple[str, str, str]] = []
    pii_total = 0

    for source_name, full_text in extracted_sources:
        SESSION.stats["sources_processed"] += 1
        raw_chunks = IngestionEngine.chunk_text(full_text, target_words=chunk_words)
        SESSION.stats["raw_chunks"] += len(raw_chunks)

        for r_chunk in raw_chunks:
            c_chunk = DataDebugger.pass1_clean_noise(r_chunk)
            if enable_pii:
                c_chunk, redactions = DataDebugger.pass2_scrub_pii(c_chunk)
                pii_total += redactions
            candidate_items.append((source_name, r_chunk, c_chunk))

    SESSION.stats["pii_redactions"] += pii_total

    if enable_dedup and candidate_items:
        clean_list = [item[2] for item in candidate_items]
        deduped_clean, dup_count = DataDebugger.pass3_deduplicate(clean_list)
        SESSION.stats["duplicates_removed"] += dup_count
        allowed_set = set(deduped_clean)
        filtered_candidates = []
        for item in candidate_items:
            if item[2] in allowed_set:
                filtered_candidates.append(item)
                allowed_set.remove(item[2])
        candidate_items = filtered_candidates

    passed_items: List[Tuple[str, str, str, float]] = []
    for source_name, r_chunk, c_chunk in candidate_items:
        passed, q_score = DataDebugger.pass4_quality_filter(c_chunk, min_words=min_words)
        if passed:
            passed_items.append((source_name, r_chunk, c_chunk, q_score))
        else:
            SESSION.stats["low_quality_removed"] += 1

    llm_instructions: List[Optional[str]] = [None] * len(passed_items)
    if use_gpu_llm and gpu_engine.model is not None and passed_items:
        prompts = [
            f"Write a single concise question or instruction that is directly answered by the following text:\n\n{item[2][:700]}\n\nInstruction:"
            for item in passed_items
        ]
        batch_size = 8
        generated: List[str] = []
        for i in range(0, len(prompts), batch_size):
            generated.extend(gpu_engine.generate_batch(prompts[i : i + batch_size], max_new_tokens=64))
        llm_instructions = [g if g else None for g in generated]

    new_records = []
    for idx, (source_name, r_chunk, c_chunk, q_score) in enumerate(passed_items):
        rec = DatasetSynthesizer.synthesize_record(
            raw_chunk=r_chunk,
            clean_chunk=c_chunk,
            source=source_name,
            quality_score=q_score,
            schema=schema,
            llm_generated_instruction=llm_instructions[idx],
        )
        new_records.append(rec)
        SESSION.stats["total_tokens"] += rec["tokens"]

    SESSION.records.extend(new_records)
    SESSION.stats["debugged_chunks"] = len(SESSION.records)
    SESSION.stats["last_duration_sec"] = round(time.time() - start_time, 2)

    return {
        "added_records": len(new_records),
        "total_records": len(SESSION.records),
        "stats": SESSION.stats,
        "errors": errors,
        "preview": SESSION.records[-50:][::-1],
    }


@app.get("/api/export/{fmt}")
async def api_export(fmt: str):
    if not SESSION.records:
        raise HTTPException(status_code=400, detail="No debugged data available to export.")

    if fmt == "jsonl":
        lines = [json.dumps(r["formatted"], ensure_ascii=False) for r in SESSION.records]
        content = "\n".join(lines) + "\n"
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=debugged_training_data.jsonl"},
        )

    elif fmt == "csv":
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["id", "source", "quality_score", "tokens", "clean_text", "formatted_json"])
        for r in SESSION.records:
            writer.writerow([
                r["id"],
                r["source"],
                r["quality_score"],
                r["tokens"],
                r["clean_text"],
                json.dumps(r["formatted"], ensure_ascii=False),
            ])
        return StreamingResponse(
            io.BytesIO(out.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=debugged_training_data.csv"},
        )

    elif fmt == "parquet":
        if not PANDAS_AVAILABLE:
            raise HTTPException(status_code=400, detail="Install 'pandas' and 'pyarrow' for Parquet export.")
        df = pd.DataFrame([
            {
                "id": r["id"],
                "source": r["source"],
                "quality_score": r["quality_score"],
                "tokens": r["tokens"],
                "clean_text": r["clean_text"],
                "formatted_payload": json.dumps(r["formatted"], ensure_ascii=False),
            }
            for r in SESSION.records
        ])
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=debugged_training_data.parquet"},
        )

    raise HTTPException(status_code=400, detail="Unsupported export format.")


# =====================================================================
# 6. EMBEDDED FRONTEND UI
# =====================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DataForge GPU - AI Training Data Debugger</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: Inter, system-ui, sans-serif; }
    .glass { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="min-h-screen p-6">
  <div class="max-w-7xl mx-auto space-y-6">

    <header class="glass rounded-2xl p-5 flex flex-wrap items-center justify-between gap-4">
      <div>
        <h1 class="text-2xl font-bold bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent">
          &#9889; DataForge GPU: AI Training Data Debugger
        </h1>
        <p class="text-xs text-slate-400 mt-1">
          Uncapped Batch File Ingestion &bull; Multi-Link Web Scraper &bull; 4-Pass Sanitization &bull; LLM Fine-Tuning Synthesizer
        </p>
      </div>
      <div class="flex flex-wrap items-center gap-3 text-xs">
        <div class="px-3 py-2 rounded-xl bg-slate-900 border border-slate-800">
          <span class="text-slate-400">Compute Engine:</span>
          <span id="gpuName" class="font-semibold text-emerald-400 ml-1">Detecting...</span>
        </div>
        <div class="px-3 py-2 rounded-xl bg-slate-900 border border-slate-800">
          <span class="text-slate-400">VRAM:</span>
          <span id="gpuVram" class="font-semibold text-cyan-400 ml-1">0 / 0 GB</span>
        </div>
        <button onclick="clearDataset()" class="px-3 py-2 rounded-xl bg-rose-950/60 hover:bg-rose-900 text-rose-300 border border-rose-800/50 transition">
          Reset Dataset
        </button>
      </div>
    </header>

    <section class="grid grid-cols-2 md:grid-cols-6 gap-4">
      <div class="glass p-4 rounded-xl">
        <div class="text-xs text-slate-400">Sources Processed</div>
        <div id="statSources" class="text-2xl font-bold text-white mt-1">0</div>
      </div>
      <div class="glass p-4 rounded-xl">
        <div class="text-xs text-slate-400">Raw Chunks</div>
        <div id="statRaw" class="text-2xl font-bold text-slate-300 mt-1">0</div>
      </div>
      <div class="glass p-4 rounded-xl">
        <div class="text-xs text-slate-400">Debugged &amp; Ready</div>
        <div id="statDebugged" class="text-2xl font-bold text-emerald-400 mt-1">0</div>
      </div>
      <div class="glass p-4 rounded-xl">
        <div class="text-xs text-slate-400">Duplicates Cut</div>
        <div id="statDups" class="text-2xl font-bold text-amber-400 mt-1">0</div>
      </div>
      <div class="glass p-4 rounded-xl">
        <div class="text-xs text-slate-400">PII Scrubbed</div>
        <div id="statPii" class="text-2xl font-bold text-purple-400 mt-1">0</div>
      </div>
      <div class="glass p-4 rounded-xl">
        <div class="text-xs text-slate-400">Trainable Tokens</div>
        <div id="statTokens" class="text-2xl font-bold text-cyan-400 mt-1">0</div>
      </div>
    </section>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2 space-y-6">
        <div class="glass rounded-2xl p-5 space-y-4">
          <h2 class="text-sm font-semibold uppercase tracking-wider text-indigo-400">1. Uncapped Multi-File &amp; Web Link Ingestion</h2>

          <label class="block border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 text-center cursor-pointer transition bg-slate-900/40">
            <input id="fileInput" type="file" multiple class="hidden" onchange="updateFileCount()" />
            <div class="text-sm font-medium text-slate-200">Drag &amp; Drop Unlimited Files Here (PDF, TXT, CSV, JSON, JSONL, MD, HTML)</div>
            <div id="fileCountLabel" class="text-xs text-indigo-400 mt-2">No file cap &mdash; select 1 or 1,000+ files at once</div>
          </label>

          <div>
            <label class="text-xs text-slate-400 block mb-1">Online Links / Chrome &amp; Firefox URLs / XML Sitemaps (One URL per line):</label>
            <textarea id="urlInput" rows="3" placeholder="https://en.wikipedia.org/wiki/Large_language_model&#10;https://example.com/sitemap.xml"
              class="w-full rounded-xl bg-slate-900 border border-slate-800 p-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"></textarea>
          </div>

          <div>
            <label class="text-xs text-slate-400 block mb-1">Direct Raw Text / Unstructured Dump (Optional):</label>
            <textarea id="rawTextInput" rows="3" placeholder="Paste raw unformatted text, logs, or transcripts here..."
              class="w-full rounded-xl bg-slate-900 border border-slate-800 p-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"></textarea>
          </div>
        </div>
      </div>

      <div class="space-y-6">
        <div class="glass rounded-2xl p-5 space-y-4">
          <h2 class="text-sm font-semibold uppercase tracking-wider text-cyan-400">2. Debugging &amp; Training Schema</h2>

          <div>
            <label class="text-xs text-slate-400 block mb-1">Target AI Training Format:</label>
            <select id="schemaSelect" class="w-full rounded-xl bg-slate-900 border border-slate-800 p-2.5 text-sm text-white">
              <option value="alpaca">Alpaca Instruction (instruction / input / output)</option>
              <option value="chatml">OpenAI ChatML / Unsloth (messages array)</option>
              <option value="sharegpt">ShareGPT Multi-Turn (conversations array)</option>
              <option value="dpo">DPO Preference Tuning (prompt / chosen / rejected)</option>
              <option value="pretrain">Raw Causal Pretraining (text + metadata)</option>
            </select>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="text-xs text-slate-400 block mb-1">Words per Chunk:</label>
              <input id="chunkWords" type="number" value="220" min="50" max="2000"
                class="w-full rounded-xl bg-slate-900 border border-slate-800 p-2 text-sm text-white" />
            </div>
            <div>
              <label class="text-xs text-slate-400 block mb-1">Min Quality Words:</label>
              <input id="minWords" type="number" value="20" min="5" max="500"
                class="w-full rounded-xl bg-slate-900 border border-slate-800 p-2 text-sm text-white" />
            </div>
          </div>

          <div class="space-y-2 pt-1 text-xs">
            <label class="flex items-center gap-2 cursor-pointer">
              <input id="enableDedup" type="checkbox" checked class="rounded bg-slate-900 border-slate-700 text-indigo-500" />
              <span>MinHash N-Gram Deduplication (Pass 2)</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer">
              <input id="enablePii" type="checkbox" checked class="rounded bg-slate-900 border-slate-700 text-indigo-500" />
              <span>Scrub Emails, Phones, IPs &amp; API Keys (Pass 3)</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer">
              <input id="useGpuLlm" type="checkbox" class="rounded bg-slate-900 border-slate-700 text-indigo-500" />
              <span>Use Local GPU HF Model for Prompt Synthesis</span>
            </label>
          </div>

          <div class="pt-2 border-t border-slate-800/80">
            <div class="flex gap-2">
              <input id="modelIdInput" type="text" value="Qwen/Qwen2.5-1.5B-Instruct"
                class="flex-1 rounded-xl bg-slate-900 border border-slate-800 px-3 py-1.5 text-xs text-slate-300" />
              <button onclick="loadGpuModel()" id="loadModelBtn"
                class="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-cyan-300 border border-slate-700 transition">
                Load on GPU
              </button>
            </div>
          </div>

          <button onclick="runPipeline()" id="processBtn"
            class="w-full py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white shadow-lg shadow-indigo-950/50 transition">
            &#9889; Debug &amp; Synthesize Training Data
          </button>

          <div class="pt-3 border-t border-slate-800">
            <div class="text-xs text-slate-400 mb-2">3. One-Click Dataset Export:</div>
            <div class="grid grid-cols-3 gap-2">
              <a href="/api/export/jsonl" class="text-center py-2 rounded-xl bg-emerald-950/70 hover:bg-emerald-900 border border-emerald-800/60 text-emerald-300 text-xs font-semibold transition">
                &#8681; .JSONL
              </a>
              <a href="/api/export/parquet" class="text-center py-2 rounded-xl bg-cyan-950/70 hover:bg-cyan-900 border border-cyan-800/60 text-cyan-300 text-xs font-semibold transition">
                &#8681; .Parquet
              </a>
              <a href="/api/export/csv" class="text-center py-2 rounded-xl bg-indigo-950/70 hover:bg-indigo-900 border border-indigo-800/60 text-indigo-300 text-xs font-semibold transition">
                &#8681; .CSV
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="alertBox" class="hidden glass rounded-xl p-4 border-l-4 border-amber-500 text-xs text-amber-300"></div>

    <section class="glass rounded-2xl p-5 space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-emerald-400">
          Side-by-Side Debug Inspector (Raw Input vs. Debugged Trainable Payload)
        </h2>
        <span id="durationBadge" class="text-xs text-slate-400"></span>
      </div>
      <div id="inspectorContainer" class="space-y-3 max-h-[520px] overflow-y-auto pr-1">
        <div class="text-center text-slate-500 text-sm py-12">
          Upload files, paste browser URLs, or enter text above and click "Debug &amp; Synthesize" to inspect live training pairs.
        </div>
      </div>
    </section>

  </div>

  <script>
    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      document.getElementById('gpuName').textContent = data.gpu.device_name + ' (' + data.gpu.device.toUpperCase() + ')';
      document.getElementById('gpuVram').textContent = data.gpu.vram_allocated_gb + ' / ' + data.gpu.vram_total_gb + ' GB';
      updateStatsUI(data.stats);
    }

    function updateStatsUI(s) {
      document.getElementById('statSources').textContent = s.sources_processed;
      document.getElementById('statRaw').textContent = s.raw_chunks;
      document.getElementById('statDebugged').textContent = s.debugged_chunks;
      document.getElementById('statDups').textContent = s.duplicates_removed;
      document.getElementById('statPii').textContent = s.pii_redactions;
      document.getElementById('statTokens').textContent = s.total_tokens.toLocaleString();
      if (s.last_duration_sec) {
        document.getElementById('durationBadge').textContent = `Processed in ${s.last_duration_sec}s`;
      }
    }

    function updateFileCount() {
      const files = document.getElementById('fileInput').files;
      const lbl = document.getElementById('fileCountLabel');
      lbl.textContent = files.length > 0 ? `${files.length} file(s) queued for GPU batch ingestion` : 'No file cap - select 1 or 1,000+ files at once';
    }

    async function loadGpuModel() {
      const btn = document.getElementById('loadModelBtn');
      const modelId = document.getElementById('modelIdInput').value;
      btn.textContent = 'Loading...';
      btn.disabled = true;
      try {
        const fd = new FormData();
        fd.append('model_id', modelId);
        const res = await fetch('/api/load-model', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to load model');
        document.getElementById('useGpuLlm').checked = true;
        await refreshStatus();
        alert('Model loaded onto ' + data.gpu.device_name);
      } catch (e) {
        alert(e.message);
      } finally {
        btn.textContent = 'Load on GPU';
        btn.disabled = false;
      }
    }

    async function runPipeline() {
      const btn = document.getElementById('processBtn');
      const alertBox = document.getElementById('alertBox');
      alertBox.classList.add('hidden');
      btn.textContent = 'Debugging & Synthesizing Batch...';
      btn.disabled = true;

      try {
        const fd = new FormData();
        const files = document.getElementById('fileInput').files;
        for (let i = 0; i < files.length; i++) {
          fd.append('files', files[i]);
        }
        fd.append('urls', document.getElementById('urlInput').value);
        fd.append('raw_text', document.getElementById('rawTextInput').value);
        fd.append('schema', document.getElementById('schemaSelect').value);
        fd.append('chunk_words', document.getElementById('chunkWords').value);
        fd.append('min_words', document.getElementById('minWords').value);
        fd.append('enable_dedup', document.getElementById('enableDedup').checked);
        fd.append('enable_pii', document.getElementById('enablePii').checked);
        fd.append('use_gpu_llm', document.getElementById('useGpuLlm').checked);

        const res = await fetch('/api/process', { method: 'POST', body: fd });
        const data = await res.json();
        updateStatsUI(data.stats);
