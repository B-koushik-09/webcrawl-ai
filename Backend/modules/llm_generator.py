"""
CollegeWeb AI — LLM Answer Generator
3-Tier Fallback Chain
  1. Gemini 2.0 Flash    (Google AI — fastest, best quality, free tier)
  2. HF Mistral 7B       (Hugging Face Inference Router → Serverless API)
  3. Ollama qwen2.5:1.5b (Local — offline / demo safety net)

Each tier is tried in order. If all fail, a retrieval-only answer is returned.
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Ensure the Backend directory is on sys.path so 'config' is importable
# whether this file is run directly or imported as a module.
_BACKEND_DIR = Path(__file__).parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── Optional imports ───────────────────────────────────────────────────────
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config import LLM_CONFIG


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    answer: str
    model: str
    provider: str
    citations_used: List[str]
    generation_time: float


# ── LLM Generator ──────────────────────────────────────────────────────────

class LLMGenerator:
    """
    Unified LLM generator with 3-tier fallback.

    Priority:
      1. Gemini 2.0 Flash   (GEMINI_API_KEY)
      2. HF Mistral 7B      (HF_TOKEN)  — Router first, Serverless second
      3. Ollama local       (no key required)
    """

    # ── System prompt shared across all providers ──────────────────────────
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant for VNR Vignana Jyothi Institute of Engineering "
        "and Technology (VNRVJIET).\n\n"
        "RULES:\n"
        "1. Answer ONLY using information from the CONTEXT provided. Do NOT use outside knowledge.\n"
        "2. If the context does not contain the answer, say you couldn't find specific details and "
        "suggest contacting the relevant college department.\n"
        "3. Provide a detailed, well-written paragraph. Include all relevant facts, numbers, names.\n"
        "4. Quote exact numbers, names, dates, and facts directly from the context.\n"
        "5. Use bullet points when the question asks about multiple items.\n"
        "6. Always cite sources at the end like [Source: Page Title].\n"
        "7. Do NOT make up or infer information not in the context.\n"
        "8. Do NOT confuse table row numbers (S.No) with counts of items.\n"
        "9. For fee questions: prioritize Domestic fees unless asked about NRI/International."
    )

    # Mistral instruct wrapper (used for Serverless Inference API path)
    _MISTRAL_TMPL = "<s>[INST] {system}\n\nCONTEXT:\n{context}\n\nQUESTION: {query} [/INST]"

    # ── Init ───────────────────────────────────────────────────────────────

    def __init__(self, model_name: Optional[str] = None):
        self.gemini_key  = os.getenv("GEMINI_API_KEY")
        self.hf_token    = os.getenv("HF_TOKEN")
        self.ollama_url  = os.getenv("OLLAMA_URL", "http://localhost:11434")

        cfg = LLM_CONFIG
        self.hf_model    = model_name or cfg.get("hf_model",     "mistralai/Mistral-7B-Instruct-v0.2")
        self.gemini_model= cfg.get("gemini_model", "gemini-2.0-flash")
        self.ollama_model= cfg.get("ollama_model", "qwen2.5:1.5b")
        self.temperature = cfg.get("temperature",  0.1)
        self.max_tokens  = cfg.get("max_tokens",   1024)

        # Keep a .model_name alias (used by RAG engine for logging)
        self.model_name  = self.gemini_model

        self._hf_client              = None
        self._gemini_mdl             = None
        self._gemini_client          = None
        self._gemini_quota_exhausted = False  # set True on 429; skip for rest of session

        self._init_gemini()
        self._init_hf_router()

    def _init_gemini(self):
        if not GEMINI_AVAILABLE:
            print("[LLM] google-genai not installed — run: pip install google-genai")
            return
        if not self.gemini_key:
            print("[LLM] GEMINI_API_KEY not set — Gemini disabled")
            return
        try:
            self._gemini_client = google_genai.Client(api_key=self.gemini_key)
            self._gemini_mdl = True  # flag: client is ready
            print(f"[LLM] ✅ Gemini ready → {self.gemini_model}")
        except Exception as e:
            print(f"[LLM] Gemini init failed: {e}")
            self._gemini_mdl = None
            self._gemini_client = None

    def _init_hf_router(self):
        if not OPENAI_AVAILABLE or not self.hf_token:
            return
        try:
            self._hf_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.hf_token,
            )
            print(f"[LLM] ✅ HF Router ready → {self.hf_model}")
        except Exception as e:
            print(f"[LLM] HF Router init failed: {e}")
            self._hf_client = None

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> GeneratedAnswer:
        t0 = time.time()
        context_text, citations = self._build_context(context_chunks)

        errors = []

        # ── Tier 1: Gemini Flash ───────────────────────────────────────────
        if self._gemini_mdl and not self._gemini_quota_exhausted:
            try:
                answer = self._call_gemini(query, context_text)
                if answer:
                    print(f"[LLM] ✅ Answered by Gemini in {time.time()-t0:.1f}s")
                    return GeneratedAnswer(
                        answer=self._clean(answer),
                        model=self.gemini_model,
                        provider="gemini",
                        citations_used=citations,
                        generation_time=time.time() - t0,
                    )
            except Exception as e:
                err = f"Gemini: {e}"
                errors.append(err)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    self._gemini_quota_exhausted = True
                    print("[LLM] ⚠ Gemini quota exhausted — using HF Mistral for this session")
                else:
                    print(f"[LLM] ⚠ {err} — trying HF Mistral...")

        # ── Tier 2a: HF Router (Mistral) ─────────────────────────────────
        if self._hf_client:
            try:
                answer = self._call_hf_router(query, context_text)
                if answer:
                    print(f"[LLM] ✅ Answered by HF Router (Mistral) in {time.time()-t0:.1f}s")
                    return GeneratedAnswer(
                        answer=self._clean(answer),
                        model=self.hf_model,
                        provider="huggingface-router",
                        citations_used=citations,
                        generation_time=time.time() - t0,
                    )
            except Exception as e:
                err = f"HF Router: {e}"
                errors.append(err)
                print(f"[LLM] ⚠ {err} — trying HF Serverless API...")

        # ── Tier 2b: HF Serverless Inference API (Mistral) ───────────────
        if self.hf_token:
            try:
                answer = self._call_hf_serverless(query, context_text)
                if answer:
                    print(f"[LLM] ✅ Answered by HF Serverless (Mistral) in {time.time()-t0:.1f}s")
                    return GeneratedAnswer(
                        answer=self._clean(answer),
                        model=self.hf_model,
                        provider="huggingface-serverless",
                        citations_used=citations,
                        generation_time=time.time() - t0,
                    )
            except Exception as e:
                err = f"HF Serverless: {e}"
                errors.append(err)
                print(f"[LLM] ⚠ {err} — trying Ollama...")

        # ── Tier 3: Ollama (local) ────────────────────────────────────────
        try:
            answer = self._call_ollama(query, context_text)
            if answer:
                print(f"[LLM] ✅ Answered by Ollama ({self.ollama_model}) in {time.time()-t0:.1f}s")
                return GeneratedAnswer(
                    answer=self._clean(answer),
                    model=self.ollama_model,
                    provider="ollama",
                    citations_used=citations,
                    generation_time=time.time() - t0,
                )
        except Exception as e:
            err = f"Ollama: {e}"
            errors.append(err)
            print(f"[LLM] ⚠ {err} — all providers exhausted")

        # ── All failed: retrieval-only answer ─────────────────────────────
        print(f"[LLM] ❌ All LLM providers failed: {'; '.join(errors)}")
        if context_chunks:
            fallback = (
                "Here is the most relevant information I found:\n\n"
                + "\n\n".join(
                    f"• {c.get('content','').strip()[:400]}"
                    for c in context_chunks[:3]
                )
            )
        else:
            fallback = "I couldn't generate an answer. Please contact the college directly for accurate information."

        return GeneratedAnswer(
            answer=fallback,
            model="retrieval-only",
            provider="fallback",
            citations_used=citations,
            generation_time=time.time() - t0,
        )

    def is_available(self) -> bool:
        """True if at least one provider is configured."""
        return bool(self._gemini_mdl or self.hf_token or True)  # Ollama always last resort

    # ── Provider implementations ───────────────────────────────────────────

    def _call_gemini(self, query: str, context: str) -> str:
        """Google Gemini 2.0 Flash via new google-genai SDK."""
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}"
        )
        response = self._gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config=google_genai.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text

    def _call_hf_router(self, query: str, context: str) -> str:
        """Hugging Face Inference Router via OpenAI-compatible SDK."""
        completion = self._hf_client.chat.completions.create(
            model=self.hf_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content

    def _call_hf_serverless(self, query: str, context: str) -> str:
        """HF Serverless Inference API — direct requests.post (free tier)."""
        prompt = self._MISTRAL_TMPL.format(
            system=self.SYSTEM_PROMPT,
            context=context,
            query=query,
        )
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True,
                    "return_full_text": False,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        if isinstance(data, dict):
            return data.get("generated_text", "")
        raise ValueError(f"Unexpected response: {data}")

    def _call_ollama(self, query: str, context: str) -> str:
        """Ollama local inference — no API key required."""
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": (
                    f"{self.SYSTEM_PROMPT}\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {query}\n\nANSWER:"
                ),
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_context(self, chunks: List[Dict]) -> tuple:
        parts, citations = [], []
        total = 0
        MAX = 12000  # chars — fits Mistral 7B + Gemini comfortably
        for chunk in chunks:
            content = chunk.get("content", "")
            if total + len(content) > MAX:
                break
            src = chunk.get("source_name", "Unknown")
            parts.append(f"Source: {src}\nContent: {content}")
            if src not in citations:
                citations.append(src)
            total += len(content)
        return "\n\n".join(parts), citations

    @staticmethod
    def _clean(text: str) -> str:
        for ch in ("\u202f", "\u00a0", "\u2009"):
            text = text.replace(ch, " ")
        return text.strip()


# ── Singleton ──────────────────────────────────────────────────────────────

_instance: Optional[LLMGenerator] = None


def get_llm_generator() -> LLMGenerator:
    global _instance
    if _instance is None:
        _instance = LLMGenerator()
    return _instance


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    gen = LLMGenerator()
    print(f"\nGemini  : {'✅' if gen._gemini_mdl else '❌'} ({gen.gemini_model})")
    print(f"HF Token: {'✅' if gen.hf_token else '❌'} ({gen.hf_model})")
    print(f"Ollama  : 🔄 {gen.ollama_url} ({gen.ollama_model})")

    ctx = [
        {
            "content": "B.Tech tuition fee at VNRVJIET is Rs 1,30,000 per year for domestic students.",
            "source_name": "Fee Structure",
        }
    ]
    res = gen.generate_answer("What is the B.Tech fee?", ctx)
    print(f"\nProvider : {res.provider}")
    print(f"Model    : {res.model}")
    print(f"Time     : {res.generation_time:.2f}s")
    print(f"Answer   : {res.answer[:300]}")
