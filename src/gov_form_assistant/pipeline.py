from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gov_form_assistant.document_io import extract_pdf_text_pages, load_document
from gov_form_assistant.field_extraction import ExtractedFields, extract_fields
from gov_form_assistant.form_understanding import FormAnalysis, analyze_form_text
from gov_form_assistant.ocr import FormOcr
from gov_form_assistant.preprocessing import NLPFeatures, preprocess_document
from gov_form_assistant.qa import FormQA
from gov_form_assistant.translation import IndicTranslator, detect_source_language


@dataclass
class PipelineResult:
    ocr_text: str
    analysis: FormAnalysis
    fields: ExtractedFields
    nlp_features: NLPFeatures
    translations: dict[str, str] = field(default_factory=dict)
    simplified: str = ""
    ocr_confidence: float | None = None
    ocr_lines: list[dict[str, float | str]] = field(default_factory=list)


def _has_usable_pdf_text(text: str) -> bool:
    return sum(character.isalnum() for character in text) >= 60


def load_project_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class GovFormPipeline:
    def __init__(self, config_path: Path, use_gpu: bool = False, lazy: bool = True):
        self.root = config_path.parent.resolve()
        self.cfg = load_project_config(config_path)
        self.cache_dir = str(self.root / self.cfg["paths"]["cache_dir"])
        self.use_gpu = use_gpu
        self._ocr: FormOcr | None = None
        self._translator: IndicTranslator | None = None
        self._qa: FormQA | None = None
        if not lazy:
            self._ensure_models()

    def _ensure_models(self, source_lang: str = "auto") -> None:
        ocr_cfg = self.cfg["models"].get("ocr", {})
        configured_languages = ocr_cfg.get("languages", ["en", "hi"])
        if source_lang in ("hi", "mr"):
            desired_languages = list(dict.fromkeys(["en", source_lang]))
        elif source_lang == "en":
            desired_languages = ["en"]
        else:
            desired_languages = list(dict.fromkeys([*configured_languages, "mr"]))
        if self._ocr is None or self._ocr.languages != desired_languages:
            self._ocr = FormOcr(
                languages=desired_languages,
                gpu=self.use_gpu,
                min_confidence=ocr_cfg.get("min_confidence", 0.2),
            )
        if self._translator is None:
            m = self.cfg["models"]["translation"]
            finetuned = m.get("finetuned_local")
            finetuned_path = str(self.root / finetuned) if finetuned else None
            if finetuned_path and not Path(finetuned_path).is_dir():
                finetuned_path = None
            self._translator = IndicTranslator(
                m["alt_indic"],
                m["id"],
                self.cache_dir,
                fallback_model_id=m.get("fallback_public", "facebook/m2m100_418M"),
                glossary_path=str(self.root / "data" / "glossary" / "government_forms.json"),
                finetuned_local_dir=finetuned_path,
                local_files_only=m.get("local_files_only", False),
            )
        if self._qa is None:
            qa_cfg = self.cfg["models"]["qa_explanation"]
            ollama_cfg = qa_cfg.get("ollama", {})
            self._qa = FormQA(
                qa_cfg["id"],
                self.cache_dir,
                ollama_model=ollama_cfg.get("model"),
                ollama_base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
                ollama_timeout_seconds=ollama_cfg.get("timeout_seconds", 180),
                ollama_context_chars=ollama_cfg.get("context_chars", 12000),
            )

    def process_file(
        self,
        path: Path,
        target_lang: str = "en",
        simplify: bool = True,
        source_lang: str = "auto",
    ) -> PipelineResult:
        self._ensure_models(source_lang)
        assert self._ocr and self._translator and self._qa

        ocr_confidences: list[float] = []
        ocr_lines: list[dict[str, float | str]] = []
        if path.suffix.lower() == ".pdf":
            native_pages = extract_pdf_text_pages(path)
            if native_pages and all(_has_usable_pdf_text(text) for text in native_pages):
                page_texts = native_pages
            else:
                pages = load_document(path)
                page_texts = []
                for index, image in enumerate(pages):
                    if index < len(native_pages) and _has_usable_pdf_text(native_pages[index]):
                        page_texts.append(native_pages[index])
                    else:
                        page_text, confidence, details = self._ocr.read_with_confidence(image)
                        page_texts.append(page_text)
                        ocr_lines.extend(details)
                        if confidence is not None:
                            ocr_confidences.append(confidence)
        else:
            pages = load_document(path)
            page_texts = []
            for image in pages:
                page_text, confidence, details = self._ocr.read_with_confidence(image)
                page_texts.append(page_text)
                ocr_lines.extend(details)
                if confidence is not None:
                    ocr_confidences.append(confidence)
        nlp_features = preprocess_document("\n\n".join(page_texts))
        ocr_text = nlp_features.normalized_text

        analysis = analyze_form_text(ocr_text, project_root=self.root)
        fields = extract_fields(ocr_text)

        resolved_source = detect_source_language(ocr_text) if source_lang == "auto" else source_lang
        if resolved_source not in ("en", "hi", "mr"):
            raise ValueError(f"Unsupported source language: {resolved_source}")
        # Always generate English translation for Flan-T5 and internal reasoning
        en_text = (
            ocr_text
            if resolved_source == "en"
            else self._translator.translate_document(
                ocr_text, source_lang=resolved_source, target_lang="en"
            )
        )
        translations = {"en": en_text}

        if target_lang != "en":
            if resolved_source == target_lang:
                display = ocr_text
            else:
                display = self._translator.translate_document(
                    ocr_text, source_lang=resolved_source, target_lang=target_lang
                )
            translations[target_lang] = display
        else:
            display = en_text

        # Translate instructions and required documents to target_lang for consistent UI display
        if resolved_source != target_lang:
            if analysis.instructions:
                analysis.instructions = [
                    self._translator.translate(item, resolved_source, target_lang)
                    for item in analysis.instructions
                ]
            if analysis.required_documents:
                analysis.required_documents = [
                    self._translator.translate(item, resolved_source, target_lang)
                    for item in analysis.required_documents
                ]

        simplified = ""
        if simplify and analysis.instructions:
            blob = "\n".join(analysis.instructions)
            language_label = {"en": "English", "hi": "Hindi", "mr": "Marathi"}.get(target_lang, "English")
            generated_summary = self._qa.simplify(blob, target_lang_label=language_label)
            if generated_summary and "following government form instructions" not in generated_summary.lower():
                simplified = generated_summary
            else:
                simplified = blob

        analysis.simplified_summary = simplified
        return PipelineResult(
            ocr_text=ocr_text,
            analysis=analysis,
            fields=fields,
            nlp_features=nlp_features,
            translations=translations,
            simplified=simplified,
            ocr_confidence=(sum(ocr_confidences) / len(ocr_confidences)) if ocr_confidences else None,
            ocr_lines=ocr_lines,
        )

    def ask(self, question: str, context: str) -> str:
        self._ensure_models()
        assert self._qa
        return self._qa.answer(question, context)

    def ask_with_evidence(
        self,
        question: str,
        context: str,
        ocr_confidence: float | None = None,
        en_context: str | None = None,
        analysis: FormAnalysis | None = None,
        fields: ExtractedFields | None = None,
        target_lang: str = "en",
    ) -> dict[str, str | list[str]]:
        self._ensure_models()
        assert self._qa
        if ocr_confidence is not None and ocr_confidence < 0.35:
            return {
                "answer": "The scanned text is unclear, so I can’t answer reliably. Please check the source or upload a clearer scan.",
                "evidence": [],
            }

        ql = question.lower().strip()

        # 1. Direct grounded answer for required supporting documents
        if any(w in ql for w in ["required doc", "documents", "supporting doc", "proof", "कागदपत्र", "दस्तावेज", "पुरावा"]):
            if analysis and analysis.required_documents:
                doc_list = "\n• " + "\n• ".join(analysis.required_documents)
                return {
                    "answer": f"The following supporting documents are required according to the form:{doc_list}",
                    "evidence": [analysis.required_documents[0]],
                }
            return {
                "answer": "No specific supporting documents are required according to the extracted form text.",
                "evidence": [],
            }

        # 2. Direct grounded answer for form filling instructions
        if any(w in ql for w in ["instruction", "how to fill", "rule", "guideline", "नियम", "सूचना", "कसे भरावे"]):
            if analysis and analysis.instructions:
                inst_list = "\n• " + "\n• ".join(analysis.instructions[:5])
                return {
                    "answer": f"Important instructions for filling this form:{inst_list}",
                    "evidence": [analysis.instructions[0]],
                }

        # 3. Document purpose / What is this form for?
        if any(w in ql for w in ["what is this form", "which form", "purpose of", "form for", "कशाचा अर्ज", "कसला फॉर्म"]):
            sample_lines = [ln.strip() for ln in (en_context or context).splitlines() if len(ln.strip()) > 3][:6]
            if sample_lines:
                title = " / ".join(sample_lines[:2])
                return {
                    "answer": f"This document appears to be: {title}",
                    "evidence": [sample_lines[0]],
                }

        # 4. Check for blank form when asking for applicant name
        if any(w in ql for w in ["applicant name", "candidate name", "person name", "who is the applicant", "अर्जदाराचे नाव"]):
            extracted_name = fields.fields.get("name") if fields else None
            if extracted_name:
                return {
                    "answer": extracted_name,
                    "evidence": [ln for ln in context.splitlines() if extracted_name in ln][:1] or [extracted_name],
                }
            has_name_label = any(
                token in context.lower()
                for token in ["अर्जदाराचे संपूर्ण नाव", "अजंदाराचे संपूर्ण नाव", "अर्जदाराचे नाव", "आवेदक का नाम", "applicant name", "full name"]
            )
            if has_name_label:
                return {
                    "answer": "This appears to be a blank application form. The 'Applicant Name' field ('अर्जदाराचे संपूर्ण नाव') is present on the form, but it has not been filled in yet.",
                    "evidence": [ln for ln in context.splitlines() if any(k in ln for k in ["अर्जदाराचे", "अजंदाराचे", "आवेदक", "Applicant"])][:1],
                }

        # 5. General Q&A: Use en_context so Flan-T5 can reason over English text
        query_context = en_context if en_context else context
        answer = self._qa.answer(question, query_context)

        if target_lang != "en" and answer and not self._qa._is_abstention(answer):
            answer = self._translator.translate(answer, "en", target_lang)

        evidence = self._qa.find_evidence(answer, question, query_context)
        if not evidence and not self._qa._is_abstention(answer):
            for line in query_context.splitlines():
                clean_line = line.strip()
                if answer.lower() in clean_line.lower() and len(clean_line) > 2:
                    evidence = [clean_line]
                    break

        return {"answer": answer, "evidence": evidence}

