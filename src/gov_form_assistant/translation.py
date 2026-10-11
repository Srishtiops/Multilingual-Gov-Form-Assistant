from __future__ import annotations

import re
import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}
M2M100_LANGS = {"en": "en", "hi": "hi", "mr": "mr"}

MARATHI_MARKERS = ("आहे", "आणि", "नाही", "तुम्ही", "साठी", "माझा", "माझे", "पैकी", "अर्जदार")
HINDI_MARKERS = ("है", "और", "नहीं", "लिए", "आपका", "करें", "किया", "अभ्यर्थी")


def detect_source_language(text: str) -> str:
    devanagari_count = len(re.findall(r"[\u0900-\u097f]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if devanagari_count < max(5, latin_count):
        return "en"
    marathi_score = sum(text.count(marker) for marker in MARATHI_MARKERS)
    hindi_score = sum(text.count(marker) for marker in HINDI_MARKERS)
    return "mr" if marathi_score > hindi_score else "hi"


@lru_cache(maxsize=4)
def _load_pair(
    model_id: str,
    cache_dir: str | None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            **kwargs,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as first_exc:
        # If online attempt failed due to network / socket / permissions, retry offline
        if not local_files_only:
            try:
                kwargs["local_files_only"] = True
                tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    **kwargs,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            except Exception:
                raise first_exc
        else:
            raise first_exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tok, model, device


class IndicTranslator:
    def __init__(
        self,
        en_to_indic_id: str,
        indic_to_en_id: str,
        cache_dir: str | None = None,
        fallback_model_id: str | None = None,
        glossary_path: str | None = None,
        finetuned_local_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.en_to_indic_id = en_to_indic_id
        self.indic_to_en_id = indic_to_en_id
        self.cache_dir = cache_dir
        self.fallback_model_id = fallback_model_id or "facebook/nllb-200-distilled-600M"
        self.finetuned_local_dir = finetuned_local_dir
        self.local_files_only = local_files_only
        self.glossary = []
        if glossary_path and Path(glossary_path).exists():
            with Path(glossary_path).open(encoding="utf-8") as handle:
                payload = json.load(handle)
            self.glossary = payload.get("terms", [])

    def _standardize_glossary(self, text: str, target_lang: str) -> str:
        if target_lang not in ("en", "hi", "mr"):
            return text
        # Sort terms by length of key in descending order to avoid partial sub-word replacement
        for term in sorted(self.glossary, key=lambda item: len(item.get("en", "")), reverse=True):
            target = term.get(target_lang)
            if not target:
                continue
            canonical_en = term.get("en")
            # Replaces English phrases if left untranslated
            if canonical_en and canonical_en.lower() != target.lower():
                text = re.sub(rf"(?i)(?<!\w){re.escape(canonical_en)}(?!\w)", target, text)

            if target_lang == "mr":
                # When target is Marathi, also convert Hindi terminology to official Marathi administrative terms
                hi_term = term.get("hi")
                if hi_term and hi_term != target and len(hi_term) > 2:
                    text = text.replace(hi_term, target)
            elif target_lang == "hi":
                # When target is Hindi, convert Marathi terminology to official Hindi terms
                mr_term = term.get("mr")
                if mr_term and mr_term != target and len(mr_term) > 2:
                    text = text.replace(mr_term, target)
            elif target_lang == "en":
                for alt_lang in ("hi", "mr"):
                    alt_term = term.get(alt_lang)
                    if alt_term and len(alt_term) > 2:
                        text = text.replace(alt_term, target)
        return text

    def _glossary_fallback(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Graceful fallback when neural models are unavailable."""
        translated: list[str] = []
        for text in texts:
            out = text
            for term in sorted(self.glossary, key=lambda item: len(item.get(src, "") or ""), reverse=True):
                source_word = term.get(src)
                target_word = term.get(tgt)
                if source_word and target_word:
                    out = re.sub(rf"(?i)(?<!\w){re.escape(source_word)}(?!\w)", target_word, out)
            translated.append(out)
        return translated

    def _active_model_id(self, preferred_id: str) -> str:
        if self.finetuned_local_dir and Path(self.finetuned_local_dir).is_dir():
            return self.finetuned_local_dir
        return preferred_id

    def _translate_batch(self, texts: list[str], src: str, tgt: str, model_id: str) -> list[str]:
        if not texts:
            return []

        model_id = self._active_model_id(model_id)
        local_only = self.local_files_only or Path(model_id).is_dir()
        try:
            if "nllb" in model_id.lower() or Path(model_id).is_dir():
                return self._translate_batch_nllb(texts, src, tgt, model_id)
            if "m2m100" in model_id.lower():
                return self._translate_batch_m2m100(texts, src, tgt, model_id)
            tok, model, device = _load_pair(
                model_id, self.cache_dir, trust_remote_code=True, local_files_only=local_only
            )
            src_code = LANG_CODES[src]
            tgt_code = LANG_CODES[tgt]
            prefixed = [f"{src_code} {tgt_code} {t}" for t in texts]
            inputs = tok(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, num_beams=4)
            return tok.batch_decode(out, skip_special_tokens=True)
        except Exception:
            if model_id != self.fallback_model_id and self.fallback_model_id:
                try:
                    return self._translate_batch(texts, src, tgt, self.fallback_model_id)
                except Exception:
                    pass
            # Final safe fallback: glossary translation instead of fatal app crash
            return self._glossary_fallback(texts, src, tgt)

    def _translate_batch_nllb(self, texts: list[str], src: str, tgt: str, model_id: str) -> list[str]:
        local_only = self.local_files_only or Path(model_id).is_dir()
        try:
            tok, model, device = _load_pair(
                model_id, self.cache_dir, trust_remote_code=False, local_files_only=local_only
            )
        except Exception:
            # Retry with explicit local_files_only
            tok, model, device = _load_pair(
                model_id, self.cache_dir, trust_remote_code=False, local_files_only=True
            )
        tok.src_lang = LANG_CODES[src]
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        max_new_tokens = min(192, max(32, int(inputs["input_ids"].shape[1] * 1.5)))
        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(LANG_CODES[tgt]),
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        return [text.strip() for text in tok.batch_decode(output, skip_special_tokens=True)]

    def _translate_batch_m2m100(self, texts: list[str], src: str, tgt: str, model_id: str) -> list[str]:
        local_only = self.local_files_only and not Path(model_id).is_dir()
        tok, model, device = _load_pair(
            model_id, self.cache_dir, trust_remote_code=False, local_files_only=local_only
        )
        src_code = M2M100_LANGS.get(src, "en")
        tgt_code = M2M100_LANGS.get(tgt, "en")
        tok.src_lang = src_code
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tok.get_lang_id(tgt_code),
                max_new_tokens=256,
                num_beams=4,
            )
        return tok.batch_decode(out, skip_special_tokens=True)

    def to_english(self, text: str, source_lang: str = "hi") -> str:
        if source_lang == "en" or not text.strip():
            return text
        chunks = self._chunk(text)
        parts = self._translate_batch(chunks, source_lang, "en", self.indic_to_en_id)
        return self._standardize_glossary("\n".join(parts), "en")

    def from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en" or not text.strip():
            return text
        chunks = self._chunk(text)
        parts = self._translate_batch(chunks, "en", target_lang, self.en_to_indic_id)
        return self._standardize_glossary("\n".join(parts), target_lang)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip() or source_lang == target_lang:
            return text

        # Exact match glossary check for form fields and certificates (100% precision, zero hallucination)
        cleaned_text = text.strip()
        for term in self.glossary:
            src_term = (term.get(source_lang) or "").strip()
            tgt_term = (term.get(target_lang) or "").strip()
            if src_term and tgt_term and cleaned_text.casefold() == src_term.casefold():
                return tgt_term

        # Check label-value pairs like "शिधापत्रिका:" or "उत्पन्नाचा दाखला: ₹50,000"
        for sep in (":", "-"):
            if sep in cleaned_text:
                parts = cleaned_text.split(sep, 1)
                label_candidate = parts[0].strip()
                val_candidate = parts[1].strip()
                for term in self.glossary:
                    src_term = (term.get(source_lang) or "").strip()
                    tgt_term = (term.get(target_lang) or "").strip()
                    if src_term and tgt_term and label_candidate.casefold() == src_term.casefold():
                        trans_val = self.translate(val_candidate, source_lang, target_lang) if val_candidate else ""
                        return f"{tgt_term}{sep} {trans_val}".strip()

        if target_lang == "en":
            return self.to_english(text, source_lang=source_lang)
        if source_lang == "en":
            return self.from_english(text, target_lang=target_lang)
        # Indic to Indic direct translation (e.g. hi -> mr or mr -> hi via NLLB)
        chunks = self._chunk(text)
        active_id = self._active_model_id(self.fallback_model_id or self.en_to_indic_id)
        parts = self._translate_batch(chunks, source_lang, target_lang, active_id)
        return self._standardize_glossary("\n".join(parts), target_lang)

    def translate_document(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translates a document while strictly preserving original line breaks, blank lines, and visual form layout."""
        if not text.strip() or source_lang == target_lang:
            return text

        raw_lines = text.splitlines()
        if not raw_lines:
            return text

        # Separate non-empty lines for batched translation
        non_empty_indices = []
        lines_to_translate = []
        for idx, line in enumerate(raw_lines):
            stripped = line.strip()
            if stripped:
                non_empty_indices.append(idx)
                lines_to_translate.append(stripped)

        if not lines_to_translate:
            return text

        # Fast batched translation
        translated_lines = []
        batch_size = 16
        for i in range(0, len(lines_to_translate), batch_size):
            chunk_lines = lines_to_translate[i : i + batch_size]
            batch_needed = []
            batch_needed_indices = []
            translated_chunk = [None] * len(chunk_lines)

            for line_i, line_text in enumerate(chunk_lines):
                # Exact glossary match check
                exact_match = None
                for term in self.glossary:
                    src_term = (term.get(source_lang) or "").strip()
                    tgt_term = (term.get(target_lang) or "").strip()
                    if src_term and tgt_term and line_text.casefold() == src_term.casefold():
                        exact_match = tgt_term
                        break
                if exact_match:
                    translated_chunk[line_i] = exact_match
                else:
                    batch_needed_indices.append(line_i)
                    batch_needed.append(line_text)

            if batch_needed:
                active_id = self._active_model_id(
                    self.fallback_model_id
                    if (source_lang != "en" and target_lang != "en")
                    else (self.indic_to_en_id if target_lang == "en" else self.en_to_indic_id)
                )
                neural_results = self._translate_batch(batch_needed, source_lang, target_lang, active_id)
                for b_idx, n_text in zip(batch_needed_indices, neural_results):
                    standardized = self._standardize_glossary(n_text, target_lang)
                    translated_chunk[b_idx] = standardized

            translated_lines.extend(translated_chunk)

        # Reassemble into original document format
        output_lines = list(raw_lines)
        for orig_idx, trans_text in zip(non_empty_indices, translated_lines):
            output_lines[orig_idx] = trans_text or ""

        return "\n".join(output_lines)

    @staticmethod
    def _chunk(text: str, max_chars: int = 400) -> list[str]:
        sentences: list[str] = []
        for line in text.splitlines() or [text]:
            for sentence in re.split(r"(?<=[.!?।॥])\s+", line.strip()):
                sentence = sentence.strip()
                if len(sentence) <= max_chars:
                    if sentence:
                        sentences.append(sentence)
                    continue
                buffer = ""
                for word in sentence.split():
                    if buffer and len(buffer) + len(word) + 1 > max_chars:
                        sentences.append(buffer)
                        buffer = ""
                    if len(word) > max_chars:
                        if buffer:
                            sentences.append(buffer)
                            buffer = ""
                        sentences.extend(word[index:index + max_chars] for index in range(0, len(word), max_chars))
                    else:
                        buffer = f"{buffer} {word}".strip()
                if buffer:
                    sentences.append(buffer)
        chunks: list[str] = []
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= max_chars:
                buf = (buf + "\n" + s).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
        return chunks or [text[:max_chars]]
