from __future__ import annotations

import re
from functools import lru_cache

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gov_form_assistant.field_extraction import extract_fields
from gov_form_assistant.preprocessing import select_relevant_context, split_sentences


@lru_cache(maxsize=1)
def _load_flan(model_id: str, cache_dir: str | None):
    kwargs = {"cache_dir": cache_dir}
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs, dtype=dtype)
    except Exception:
        kwargs["local_files_only"] = True
        tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs, dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tok, model, device


class FormQA:
    def __init__(
        self,
        model_id: str,
        cache_dir: str | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout_seconds: int = 180,
        ollama_context_chars: int = 12000,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self.ollama_context_chars = ollama_context_chars

    @staticmethod
    def _clean_answer_value(value: str) -> str:
        cleaned = value.strip().strip("-:;.,|/\\")
        while cleaned.endswith(";") or cleaned.endswith(":"):
            cleaned = cleaned[:-1].rstrip()
        return cleaned.strip()

    @staticmethod
    def _is_context_supported_answer(answer: str, context: str) -> bool:
        if not answer or not context:
            return False
        candidate = re.sub(r"[^\w\u0900-\u097f]", " ", answer).strip()
        if not candidate or len(candidate) < 2:
            return False
        lowered = candidate.lower()
        if any(token in lowered for token in [
            "not found", "not stated", "cannot determine", "not available", "cannot say", "not mentioned", "i do not know", "unanswerable"
        ]):
            return False
        normalized_context = re.sub(r"[^\w\u0900-\u097f]", " ", context).lower()
        if lowered in normalized_context:
            return True
        filler = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "and", "or", "on", "at", "by", "this", "that", "it", "be"}
        tokens = [w for w in lowered.split() if len(w) > 1 and w not in filler]
        if not tokens:
            return False
        matches = sum(1 for tok in tokens if tok in normalized_context)
        return (matches / len(tokens)) >= 0.5

    @staticmethod
    def _is_abstention(answer: str) -> bool:
        lowered = answer.casefold()
        return any(token in lowered for token in (
            "could not find", "not found", "not stated", "cannot determine", "not available", "cannot say", "not mentioned",
        ))

    @staticmethod
    def find_evidence(answer: str, question: str, context: str) -> list[str]:
        if not answer or FormQA._is_abstention(answer):
            return []
        sentences = split_sentences(context)
        answer_words = {w.casefold() for w in re.findall(r"[\w\u0900-\u097f]+", answer) if len(w) > 1}
        question_words = {w.casefold() for w in re.findall(r"[\w\u0900-\u097f]+", question) if len(w) > 2}
        ranked = []
        for sentence in sentences:
            normalized = sentence.casefold()
            if answer.casefold() in normalized:
                return [sentence.strip()]
            words = {w.casefold() for w in re.findall(r"[\w\u0900-\u097f]+", sentence) if len(w) > 1}
            score = len(words & answer_words) + len(words & question_words)
            if score:
                ranked.append((score, sentence.strip()))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            candidate = ranked[0][1]
            if FormQA._is_context_supported_answer(answer, context):
                return [candidate]
        return []

    @staticmethod
    def _looks_like_field_question(question: str) -> bool:
        ql = question.lower()
        return any(token in ql for token in [
            "name", "date of birth", "dob", "address", "phone", "mobile", "email",
            "aadhaar", "pan", "passport", "id number", "gender",
        ])

    @staticmethod
    def _has_field_value_candidate(question: str, context: str) -> bool:
        if not question or not context:
            return False
        ql = question.lower()
        patterns = []
        if "name" in ql:
            patterns.extend([r"(?:applicant|full|candidate|person)\s+name\s*[:\-]\s*(.+)", r"name\s*[:\-]\s*(.+)"])
        if "date of birth" in ql or "dob" in ql:
            patterns.extend([r"(?:date\s+of\s+birth|dob|birth\s+date)\s*[:\-]\s*(.+)"])
        if "address" in ql:
            patterns.extend([r"(?:address|residential\s+address)\s*[:\-]\s*(.+)"])
        if "phone" in ql or "mobile" in ql:
            patterns.extend([r"(?:phone|mobile|contact)\s*(?:number)?\s*[:\-]\s*(.+)"])
        if "email" in ql:
            patterns.extend([r"(?:email|e-mail)\s*(?:address)?\s*[:\-]\s*(.+)"])
        if "aadhaar" in ql or "id" in ql or "passport" in ql or "pan" in ql:
            patterns.extend([r"(?:aadhaar|pan|passport|id)\s*(?:number)?\s*[:\-]\s*(.+)"])
        if "gender" in ql:
            patterns.extend([r"gender\s*[:\-]\s*(.+)"])
        if not patterns:
            return False
        for sentence in split_sentences(context):
            if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns):
                return True
        return False

    @staticmethod
    def _extract_grounded_answer(question: str, context: str) -> str | None:
        if not question or not context:
            return None

        ql = question.lower()
        extracted = extract_fields(context)

        if any(k in ql for k in ["applicant name", "full name", "candidate name", "person name", "नाव", "नाम"]) or (
            re.search(r"\bname\b", ql) and not any(k in ql for k in ["father", "mother", "parent", "spouse", "guardian", "वडिलांचे", "आईचे", "पिता", "माता"])
        ):
            if "name" in extracted.fields:
                return extracted.fields["name"]

        if any(k in ql for k in ["parent", "father", "mother", "guardian", "spouse", "वडिलांचे", "आईचे", "पिता", "माता", "पालक"]):
            if "parent_name" in extracted.fields:
                return extracted.fields["parent_name"]

        if any(k in ql for k in ["date of birth", "dob", "birth date", "जन्मतारीख", "जन्म तारीख", "जन्म तिथि"]):
            if "date_of_birth" in extracted.fields:
                return extracted.fields["date_of_birth"]

        if any(k in ql for k in ["gender", "sex", "लिंग"]):
            if "gender" in extracted.fields:
                return extracted.fields["gender"]

        if any(k in ql for k in ["category", "caste", "reservation", "वर्ग", "प्रवर्ग", "श्रेणी"]):
            if "category" in extracted.fields:
                return extracted.fields["category"]

        if any(k in ql for k in ["enrollment", "id number", "aadhaar", "pan", "passport", "voter id", "dl number", "ओळखपत्र", "आधार", "पॅन"]):
            if "id_number" in extracted.fields:
                return extracted.fields["id_number"]

        if any(k in ql for k in ["phone", "mobile", "contact", "दूरध्वनी", "चलभाष", "मोबाइल"]):
            if "phone" in extracted.fields:
                return extracted.fields["phone"]

        if any(k in ql for k in ["email", "e-mail", "ईमेल", "ई-मेल"]):
            if "email" in extracted.fields:
                return extracted.fields["email"]

        if any(k in ql for k in ["address", "residential", "permanent address", "communication address", "पत्ता", "पता"]):
            if "address" in extracted.fields:
                return extracted.fields["address"]

        field_rules = [
            ("name", [r"(?:applicant|full|candidate|person)\s+name\s*[:\-]\s*(.+)", r"name\s*[:\-]\s*(.+)"]),
            ("date_of_birth", [r"date\s+of\s+birth\s*[:\-]\s*(.+)", r"dob\s*[:\-]\s*(.+)", r"birth\s+date\s*[:\-]\s*(.+)"]),
            ("address", [r"address\s*[:\-]\s*(.+)", r"residential\s+address\s*[:\-]\s*(.+)"]),
            ("phone", [r"phone\s*(?:number)?\s*[:\-]\s*(.+)", r"mobile\s*(?:number)?\s*[:\-]\s*(.+)", r"contact\s*(?:number)?\s*[:\-]\s*(.+)"]),
            ("email", [r"email\s*(?:address)?\s*[:\-]\s*(.+)"]),
            ("id_number", [r"(?:aadhaar|pan|passport|id|document)\s*(?:number)?\s*[:\-]\s*(.+)"]),
            ("gender", [r"gender\s*[:\-]\s*(.+)"]),
        ]

        for keywords, patterns in field_rules:
            if any(token in ql for token in {
                keywords,
                keywords.replace("_", " "),
                keywords.replace("_", ""),
            }):
                for sentence in split_sentences(context):
                    s_clean = re.sub(r"\s+", " ", sentence).strip()
                    if not s_clean:
                        continue
                    for pattern in patterns:
                        match = re.search(pattern, s_clean, flags=re.IGNORECASE)
                        if match:
                            value = FormQA._clean_answer_value(match.group(1))
                            if value and len(value) <= 200:
                                return value

        label_value_patterns = [
            r"(?:applicant|full|candidate|person)\s+name\s*[:\-]\s*(.+)",
            r"(?:date\s+of\s+birth|dob|birth\s+date)\s*[:\-]\s*(.+)",
            r"(?:address|residential\s+address)\s*[:\-]\s*(.+)",
            r"(?:phone|mobile|contact)\s*(?:number)?\s*[:\-]\s*(.+)",
            r"(?:email|e-mail)\s*(?:address)?\s*[:\-]\s*(.+)",
            r"(?:aadhaar|pan|passport|id|document)\s*(?:number)?\s*[:\-]\s*(.+)",
            r"(?:gender)\s*[:\-]\s*(.+)",
        ]

        for sentence in split_sentences(context):
            if ":" not in sentence and "-" not in sentence:
                continue
            for pattern in label_value_patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    value = FormQA._clean_answer_value(match.group(1))
                    if value and len(value) <= 200:
                        return value
        return None

    def _answer_with_ollama(self, question: str, context: str) -> str | None:
        if not self.ollama_model:
            return None

        relevant_context = select_relevant_context(question, context, self.ollama_context_chars)
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer questions about government forms using only the provided form text. "
                                "Do not guess. If the answer is absent, say it is not stated in the form. "
                                "Be concise and preserve names, dates, and document categories exactly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n\n"
                                f"Form text:\n{relevant_context}"
                            ),
                        },
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 8192},
                },
                timeout=self.ollama_timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
            return answer or None
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return None

    def answer(self, question: str, context: str, max_context_chars: int = 3000) -> str:
        if not question or not context:
            return "I could not find an answer in the extracted form text."

        relevant_context = select_relevant_context(question, context, max_context_chars)
        grounded_answer = self._extract_grounded_answer(question, relevant_context)
        if grounded_answer:
            return grounded_answer

        ollama_answer = self._answer_with_ollama(question, relevant_context)
        if ollama_answer and self._is_context_supported_answer(ollama_answer, relevant_context):
            return ollama_answer

        tok, model, device = _load_flan(self.model_id, self.cache_dir)
        prompt_prefix = (
            "You are helping an Indian citizen understand a government form. "
            "Answer briefly and clearly using only the form text. If the answer cannot be found, say 'Not stated in form'.\n"
            f"Question: {question}\nForm text:\n"
        )
        prompt_suffix = "\nAnswer:"
        prefix_tokens = tok(prompt_prefix, add_special_tokens=False)["input_ids"]
        suffix_tokens = tok(prompt_suffix, add_special_tokens=False)["input_ids"]
        context_budget = max(
            0,
            512 - len(prefix_tokens) - len(suffix_tokens) - tok.num_special_tokens_to_add(pair=False),
        )
        context_tokens = (
            tok(
                relevant_context,
                add_special_tokens=False,
                truncation=True,
                max_length=context_budget,
            )["input_ids"]
            if context_budget
            else []
        )
        ctx = tok.decode(context_tokens[:context_budget], skip_special_tokens=True)
        prompt = f"{prompt_prefix}{ctx}{prompt_suffix}"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=2)
        answer = tok.decode(out[0], skip_special_tokens=True).strip()
        if answer and not FormQA._is_abstention(answer) and self._is_context_supported_answer(answer, relevant_context):
            return answer
        return "I could not find an answer in the extracted form text."

    def simplify(self, text: str, target_lang_label: str = "English") -> str:
        prompt = (
            f"Simplify the following government form instructions in plain {target_lang_label} "
            f"for a citizen. Use short bullet points.\n\n{text}\n\nSimplified:"
        )
        tok, model, device = _load_flan(self.model_id, self.cache_dir)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, num_beams=2)
        return tok.decode(out[0], skip_special_tokens=True).strip()
