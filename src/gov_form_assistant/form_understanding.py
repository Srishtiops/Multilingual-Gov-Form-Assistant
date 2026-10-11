from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from gov_form_assistant.line_classifier import load_line_classifier

INSTRUCTION_PATTERNS = [
    r"\b(?:must|need to|required to|mandatory|please)\b",
    r"\b(?:attach|enclose|submit|upload|provide|complete|fill(?:\s+in)?|sign here|bring|pay|select)\b",
    r"कृपया|आवश्यक(?:ता)?|संलग्न|अनिवार्य|सूचना|भरा|भरावे|लिहा|जोडा|सादर करा|अपलोड करा|खूण करा|नोंद करा|दाखल करा|करावे|करा",
    r"कृपया|आवश्यक|संलग्न करें|अनिवार्य|भरें|लिखें|जमा करें|दर्ज करें|अपलोड करें|हस्ताक्षर करें|चयन करें",
]

NON_INSTRUCTION_PATTERNS = [
    r"\bshall\s+not\s+be\s+liable\b",
    r"\bdoes?\s+not\s+guarantee\b",
    r"\bnot\s+an?\s+admission\b",
    r"\blegal\s+obligations?\b",
]

REQUIREMENT_PATTERNS = [
    r"\b(?:required documents?|supporting documents?|documents? to be attached)\b",
    r"\b(?:attach|enclose|submit|upload|provide|required|mandatory)\b",
    r"\bproof of (?:identity|address|date of birth)\b",
]

DOCUMENT_PATTERNS = [
    ("Aadhaar", r"\b(?:aadhaar|aadhar|uid)\b|आधार"),
    ("PAN card", r"\bpan\b|permanent account|पैन\s*कार्ड|पॅन\s*कार्ड"),
    ("Passport", r"\bpassport\b|पासपोर्ट|पारपत्र"),
    ("Driving licence", r"driving\s*licen[cs]e|\bdl\b|ड्राइविंग\s*लाइसेंस|वाहन\s*परवाना"),
    ("Voter ID", r"voter\s*id|\bepic\b|मतदाता\s*पहचान|मतदार\s*ओळखपत्र"),
    ("Birth certificate", r"birth\s*certificate|proof of date of birth|जन्म\s*प्रमाणपत्र|जन्माचा\s*दाखला"),
    ("Proof of identity", r"proof of identity|identity proof|पहचान\s*प्रमाण|ओळखीचा\s*पुरावा"),
    ("Proof of address", r"proof of address|address\s*proof|पते\s*का\s*प्रमाण|पत्त्याचा\s*पुरावा"),
    ("Ration card", r"ration\s*card|राशन\s*कार्ड|शिधापत्रिका"),
    ("Electricity bill", r"electricity\s*bill|बिजली\s*का\s*बिल|लाईट\s*बिल"),
    ("Photograph", r"photograph|\bphoto\b|फोटो|छायाचित्र"),
    ("Affidavit", r"\baffidavit\b|शपथ\s*पत्र|प्रतिज्ञापत्र"),
    ("Self-attested copy", r"self\s*attested|स्वप्रमाणित|स्वयं\s*सत्यापित|स्वयं-साक्षांकित"),
]


@dataclass
class FormAnalysis:
    raw_text: str
    instructions: list[str] = field(default_factory=list)
    required_documents: list[str] = field(default_factory=list)
    simplified_summary: str = ""


def extract_instructions(text: str, max_items: int = 12) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    picked: list[str] = []
    for ln in lines:
        lower = ln.lower()
        if any(re.search(pattern, lower, re.I) for pattern in NON_INSTRUCTION_PATTERNS):
            continue
        if any(re.search(pattern, lower, re.I) for pattern in INSTRUCTION_PATTERNS):
            picked.append(ln)
        if len(picked) >= max_items:
            break
    return picked


def extract_required_documents(text: str) -> list[str]:
    found: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index in range(len(lines)):
        window = " ".join(lines[max(0, index - 1) : index + 2])
        if not any(re.search(pattern, window, re.I) for pattern in REQUIREMENT_PATTERNS):
            continue
        for label, pattern in DOCUMENT_PATTERNS:
            if re.search(pattern, window, re.I):
                found.append(label)
    # dedupe preserve order
    seen = set()
    out = []
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _analyze_with_indian_classifier(text: str, project_root: Path | None) -> FormAnalysis | None:
    if project_root is None:
        return None
    task = "indian_forms"
    classifier = load_line_classifier(str(project_root), task)
    if classifier is None:
        task = "form_role"
        classifier = load_line_classifier(str(project_root), task)
    if classifier is None:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    instructions: list[str] = []
    documents: list[str] = []
    for line in lines:
        label = classifier.predict(line)
        if task == "indian_forms":
            if label == "instruction" and line not in instructions:
                instructions.append(line)
            elif label == "required_document" and line not in documents:
                documents.append(line)
        else:
            if label in ("header", "question") and line not in instructions:
                instructions.append(line)
            elif label == "question" and re.search(r"\b(?:document|proof|attach)\b", line, re.I):
                documents.append(line)
    if not instructions and not documents:
        return None
    return FormAnalysis(
        raw_text=text,
        instructions=instructions[:12] or extract_instructions(text),
        required_documents=documents[:20] or extract_required_documents(text),
    )


def analyze_form_text(text: str, project_root: Path | None = None) -> FormAnalysis:
    learned = _analyze_with_indian_classifier(text, project_root)
    if learned is not None:
        return learned
    return FormAnalysis(
        raw_text=text,
        instructions=extract_instructions(text),
        required_documents=extract_required_documents(text),
    )
