"""
CollegeWeb AI — LLM Answer Generator
# 3-Tier Fallback Chain
#   1. Gemini 2.5 Flash    (Google AI — fastest, best quality, free tier)
#   2. HF Qwen2.5-72B     (Hugging Face Inference Router → Serverless API, free)
#   3. Ollama qwen2.5:1.5b (Local — offline / demo safety net)

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
    Unified LLM generator with 4-tier fallback.

    Priority:
      1. Gemini 2.0 Flash   (GEMINI_API_KEY)
      2. OpenRouter         (OPENROUTER_API_KEY)
      3. HF Qwen2.5-72B     (HF_TOKEN)  — Router first, Serverless second
      4. Ollama local       (no key required)
    """

    # ── System prompt shared across all providers ──────────────────────────
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant for VNR Vignana Jyothi Institute of Engineering "
        "and Technology (VNRVJIET).\n\n"
        "RULES:\n"
        "1. Answer ONLY using information from the CONTEXT provided. Do NOT use outside knowledge.\n"
        "2. Write the answer in your own words. Do NOT copy sentences directly from the context.\n"
        "3. Be concise and clear. Include only the facts needed to answer the question.\n"
        "4. If specific numbers, names, or dates are present, include them accurately.\n"
        "5. If the context does not contain the answer, say: "
        "\"Information not found on the website. Please contact the relevant department.\"\n"
        "6. Use bullet points only if the question asks for a list.\n"
        "7. Always cite the source at the end like [Source: Page Title].\n"
        "8. Do NOT confuse table row numbers (S.No) with counts.\n"
        "9. For fee questions: prioritize Domestic fees unless asked about NRI/International.\n"
        "10. For HOD/Head queries: only state someone is an HOD if they have 'Head' or 'HOD' "
        "in their Designation or Profile. Do not infer HOD status from just being a Professor.\n\n"
        "If your answer repeats long phrases from the context, rewrite it more concisely."
    )

    # ChatML template (used for Serverless Inference API path, works for most modern models)
    _CHATML_TMPL = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nCONTEXT:\n{context}\n\nQUESTION: {query}<|im_end|>\n<|im_start|>assistant\n"

    # ── Init ───────────────────────────────────────────────────────────────

    def __init__(self, model_name: Optional[str] = None):
        self.gemini_key      = os.getenv("GEMINI_API_KEY")
        self.openrouter_key  = os.getenv("OPENROUTER_API_KEY")
        self.hf_token        = os.getenv("HF_TOKEN")
        self.ollama_url      = os.getenv("OLLAMA_URL", "http://localhost:11434")

        cfg = LLM_CONFIG
        self.openrouter_model    = cfg.get("openrouter_model", "mistralai/mistral-7b-instruct")
        self.openrouter_fallback = cfg.get("openrouter_fallback", "upstage/solar-pro-3:free")
        self.hf_model            = model_name or cfg.get("hf_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.hf_fallback         = cfg.get("hf_fallback", "mistralai/Mistral-7B-Instruct-v0.3")
        self.gemini_model        = cfg.get("gemini_model", "gemini-2.5-flash")
        self.ollama_model        = cfg.get("ollama_model", "qwen2.5:1.5b")
        self.temperature = cfg.get("temperature",  0.1)
        self.max_tokens  = cfg.get("max_tokens",   1024)

        # Keep a .model_name alias (used by RAG engine for logging)
        self.model_name  = self.gemini_model

        self._openrouter_client      = None
        self._hf_client              = None
        self._gemini_mdl             = None
        self._gemini_client          = None
        self._gemini_quota_exhausted = False  # set True on 429; skip for rest of session

        self._init_gemini()
        self._init_openrouter()
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
            print(f"[LLM] [OK] Gemini ready -> {self.gemini_model}")
        except Exception as e:
            print(f"[LLM] Gemini init failed: {e}")
            self._gemini_mdl = None
            self._gemini_client = None

    def _init_openrouter(self):
        if not OPENAI_AVAILABLE or not self.openrouter_key:
            return
        try:
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
            )
            print(f"[LLM] [OK] OpenRouter ready -> {self.openrouter_model}")
        except Exception as e:
            print(f"[LLM] OpenRouter init failed: {e}")
            self._openrouter_client = None

    def _init_hf_router(self):
        if not OPENAI_AVAILABLE or not self.hf_token:
            return
        try:
            self._hf_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.hf_token,
            )
            print(f"[LLM] [OK] HF Router ready -> {self.hf_model}")
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
                    print(f"[LLM] [OK] Answered by Gemini in {time.time()-t0:.1f}s")
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
                    print(f"[LLM] [WARN] Gemini quota exhausted -- trying next provider")
                else:
                    print(f"[LLM] [WARN] {err} -- trying OpenRouter...")

        # ── Tier 2: OpenRouter (primary → fallback) ────────────────────────
        if self._openrouter_client:
            for or_model in [self.openrouter_model, self.openrouter_fallback]:
                try:
                    answer = self._call_openrouter(query, context_text, model=or_model)
                    if answer:
                        print(f"[LLM] [OK] Answered by OpenRouter ({or_model}) in {time.time()-t0:.1f}s")
                        return GeneratedAnswer(
                            answer=self._clean(answer),
                            model=or_model,
                            provider="openrouter",
                            citations_used=citations,
                            generation_time=time.time() - t0,
                        )
                except Exception as e:
                    err = f"OpenRouter ({or_model}): {e}"
                    errors.append(err)
                    print(f"[LLM] [WARN] {err}")
            print(f"[LLM] [WARN] All OpenRouter models failed -- trying HF...")

        # ── Tier 3: HuggingFace (primary → fallback) ───────────────────────
        hf_models = [self.hf_model, self.hf_fallback]
        for hf_model in hf_models:
            # Try HF Router first
            if self._hf_client:
                try:
                    answer = self._call_hf_router(query, context_text, model=hf_model)
                    if answer:
                        print(f"[LLM] [OK] Answered by HF Router ({hf_model}) in {time.time()-t0:.1f}s")
                        return GeneratedAnswer(
                            answer=self._clean(answer),
                            model=hf_model,
                            provider="huggingface-router",
                            citations_used=citations,
                            generation_time=time.time() - t0,
                        )
                except Exception as e:
                    err = f"HF Router ({hf_model}): {e}"
                    errors.append(err)
                    print(f"[LLM] [WARN] {err}")
            # Try HF Serverless API
            if self.hf_token:
                try:
                    answer = self._call_hf_serverless(query, context_text, model=hf_model)
                    if answer:
                        print(f"[LLM] [OK] Answered by HF Serverless ({hf_model}) in {time.time()-t0:.1f}s")
                        return GeneratedAnswer(
                            answer=self._clean(answer),
                            model=hf_model,
                            provider="huggingface-serverless",
                            citations_used=citations,
                            generation_time=time.time() - t0,
                        )
                except Exception as e:
                    err = f"HF Serverless ({hf_model}): {e}"
                    errors.append(err)
                    print(f"[LLM] [WARN] {err}")
        if hf_models:
            print(f"[LLM] [WARN] All HF models failed -- trying Ollama...")

        # ── Tier 4: Ollama (local) ────────────────────────────────────────
        try:
            answer = self._call_ollama(query, context_text)
            if answer:
                print(f"[LLM] [OK] Answered by Ollama ({self.ollama_model}) in {time.time()-t0:.1f}s")
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
            print(f"[LLM] [WARN] {err} -- all providers exhausted")

        # ── All failed: retrieval-only answer ─────────────────────────────
        print(f"[LLM] [FAIL] All LLM providers failed: {'; '.join(errors)}")
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
        return bool(self._gemini_mdl or self.openrouter_key or self.hf_token or True)  # Ollama always last resort

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

    def _call_openrouter(self, query: str, context: str, model: str = None) -> str:
        """OpenRouter via OpenAI-compatible SDK."""
        use_model = model or self.openrouter_model
        completion = self._openrouter_client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
            extra_headers={
                "HTTP-Referer": "https://vnrvjiet.ac.in",
                "X-Title": "CollegeWeb AI",
            },
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content

    def _call_hf_router(self, query: str, context: str, model: str = None) -> str:
        """Hugging Face Inference Router via OpenAI-compatible SDK."""
        use_model = model or self.hf_model
        completion = self._hf_client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content

    def _call_hf_serverless(self, query: str, context: str, model: str = None) -> str:
        """HF Serverless Inference API — /v1/chat/completions (free tier)."""
        use_model = model or self.hf_model
        url = f"https://api-inference.huggingface.co/v1/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": use_model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # /v1/chat/completions format
        if isinstance(data, dict) and "choices" in data:
            return data["choices"][0]["message"]["content"]
        raise ValueError(f"Unexpected HF response: {data}")

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
    print(f"\nGemini  : {'[OK]' if gen._gemini_mdl else '[FAIL]'} ({gen.gemini_model})")
    print(f"HF Token: {'[OK]' if gen.hf_token else '[FAIL]'} ({gen.hf_model})")
    print(f"Ollama  : [READY] {gen.ollama_url} ({gen.ollama_model})")

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
