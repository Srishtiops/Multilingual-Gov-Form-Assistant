from __future__ import annotations

import re
from dataclasses import dataclass, field


FIELD_PATTERNS: dict[str, list[str]] = {
    "name": [
        r"^\s*(?:applicant\s+name|full\s+name|candidate\s+name|name\s+of\s+the\s+applicant|name)\s*[:\-]\s*(.+)$",
        r"^\s*(?:आवेदक\s+का\s+नाम|अभ्यर्थी\s+का\s+नाम|पूरा\s+नाम|नाम)\s*[:\-]\s*(.+)$",
        r"^\s*(?:अर्जदाराचे\s+नाव|उमेदवाराचे\s+नाव|संपूर्ण\s+नाव|नाव)\s*[:\-]\s*(.+)$",
    ],
    "parent_name": [
        r"^\s*(?:parent(?:['/]guardian)?(?:['s])?\s+name|father(?:['s])?\s+name|mother(?:['s])?\s+name|spouse(?:['s])?\s+name)\s*[:\-]\s*(.+)$",
        r"^\s*(?:पिता\s+का\s+नाम|माता\s+का\s+नाम|अभिभावक\s+का\s+नाम|पति/पत्नी\s+का\s+नाम)\s*[:\-]\s*(.+)$",
        r"^\s*(?:वडिलांचे\s+नाव|आईचे\s+नाव|पालकांचे\s+नाव|पतीचे\s+नाव)\s*[:\-]\s*(.+)$",
    ],
    "date_of_birth": [
        r"^\s*(?:date\s*of\s*birth|d\.?o\.?b\.?|birth\s*date)\s*[:\-]\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
        r"^\s*(?:जन्म\s*तिथि|जन्म\s*दिनांक|जन्म\s*तारीख)\s*[:\-]\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
        r"^\s*(?:जन्मतारीख|जन्म\s*तारीख|जन्म\s*दिनांक)\s*[:\-]\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
    ],
    "gender": [
        r"^\s*(?:gender|sex)\s*[:\-]\s*(male|female|transgender|other|m|f|पुरुष|महिला|स्त्री|इतर)\b",
        r"^\s*(?:लिंग)\s*[:\-]\s*(पुरुष|महिला|स्त्री|इतर|male|female|other)\b",
    ],
    "category": [
        r"^\s*(?:category|caste|reservation\s+category)\s*[:\-]\s*([A-Za-z0-9\-+/ ]{2,30})$",
        r"^\s*(?:श्रेणी|वर्ग|जाति)\s*[:\-]\s*(.+)$",
        r"^\s*(?:प्रवर्ग|श्रेणी|जात)\s*[:\-]\s*(.+)$",
    ],
    "address": [
        r"^\s*(?:communication\s+|residential\s+|correspondence\s+|permanent\s+)?address\s*[:\-]\s*(.{10,140})$",
        r"^\s*(?:पता|स्थायी\s+पता|पत्राचार\s+का\s+पता)\s*[:\-]\s*(.{10,140})$",
        r"^\s*(?:पत्ता|कायमचा\s+पत्ता|पत्रव्यवहाराचा\s+पत्ता)\s*[:\-]\s*(.{10,140})$",
    ],
    "id_number": [
        r"^\s*(?:enrollment\s+id|aadhaar|uid|pan|passport|dl|voter|gate\s+paper\s+code)\s*(?:no\.?|number|id)?\s*[:\-]\s*([A-Z0-9\-*]{4,30})$",
        r"\b(\d{4}\s?\d{4}\s?\d{4})\b",
        r"\b([A-Z]{5}\d{4}[A-Z])\b",
    ],
    "phone": [
        r"(?:mobile|phone|contact|चलभाष|मोबाइल|दूरभाष|दूरध्वनी)\s*(?:no\.?|number|क्रमांक)?\s*[:\-]?\s*(\+?[\d\s*\-]{9,16}\d)",
        r"(\+?[\d\s*\-]{9,16}\d)\s*\((?:mobile|phone|contact|मोबाइल)\s*(?:no\.?|number)?\)",
    ],
    "email": [
        r"(?:email|e-mail|ईमेल|ई-मेल)\s*(?:address|id|पत्ता)?\s*[:\-]?\s*([\w.*+-]+@[\w.-]+\.\w+)",
        r"\b([\w.*+-]+@[\w.-]+\.\w+)\b",
    ],
}


@dataclass
class ExtractedFields:
    fields: dict[str, str] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)


def extract_fields(text: str) -> ExtractedFields:
    result = ExtractedFields()
    for field_name, patterns in FIELD_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text, re.I | re.M)
            if m:
                val = m.group(1).strip()
                val = re.sub(r"\s+", " ", val)
                if val:
                    result.fields[field_name] = val
                    break

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    adjacent_labels = {
        "name": (
            r"(?:\(?(?:full\s+name(?:\s+of\s+(?:the\s+)?applicant)?|applicant\s+name|नाम|नाव)\)?)$",
            (-1, 1),
        ),
        "parent_name": (
            r"(?:\(?(?:parent(?:['/]guardian)?(?:['s])?\s+name|father(?:['s])?\s+name|पालकांचे\s+नाव|पिता\s+का\s+नाम)\)?)$",
            (-1, 1),
        ),
        "date_of_birth": (
            r"(?:\(?(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s+date|जन्मतारीख|जन्म\s*तिथि)\)?)$",
            (-1, 1),
        ),
        "gender": (
            r"(?:\(?(?:gender|sex|लिंग)\)?)$",
            (-1, 1),
        ),
        "category": (
            r"(?:\(?(?:category|caste|वर्ग|प्रवर्ग)\)?)$",
            (-1, 1),
        ),
        "id_number": (
            r"(?:\(?(?:enrollment\s+id|aadhaar|uid|pan|passport|voter\s+id|ओळखपत्र|पहचान\s*पत्र)\)?)$",
            (-1, 1),
        ),
        "phone": (
            r"(?:\(?(?:mobile|phone|contact\s*no|मोबाइल)\)?)$",
            (-1, 1),
        ),
    }

    value_patterns = {
        "name": r"^[^\W\d_][\w .'-]{1,79}$",
        "parent_name": r"^[^\W\d_][\w .'-]{1,79}$",
        "date_of_birth": r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$",
        "gender": r"^(?:Male|Female|Transgender|Other|पुरुष|महिला|स्त्री|इतर)$",
        "category": r"^[A-Z0-9\-+/ ]{2,20}$",
        "id_number": r"^[A-Z0-9*\-]{4,30}$",
        "phone": r"^\+?[\d\s*\-]{9,18}$",
    }

    for field_name, (label_pattern, offsets) in adjacent_labels.items():
        if field_name in result.fields:
            continue
        for index, line in enumerate(lines):
            if not re.search(label_pattern, line, re.I):
                continue
            for offset in offsets:
                candidate_index = index + offset
                if not 0 <= candidate_index < len(lines):
                    continue
                candidate = lines[candidate_index].strip(" :.-")
                # Do not accept another label line as value
                if candidate.startswith("(") and candidate.endswith(")"):
                    continue
                if re.fullmatch(value_patterns[field_name], candidate, re.I):
                    result.fields[field_name] = candidate
                    break
            if field_name in result.fields:
                break

    # Multi-line address extraction if address field is still missing or short
    if "address" not in result.fields:
        for idx, line in enumerate(lines):
            if re.search(r"^(?:communication\s+|residential\s+|correspondence\s+|permanent\s+)?address\b|पत्ता\b|पता\b", line, re.I):
                address_parts = []
                for sub_idx in range(idx + 1, min(len(lines), idx + 5)):
                    sub_line = lines[sub_idx]
                    if re.search(r"^(?:mobile|phone|contact|email|d\.?o\.?b|gender|category|qualification)\b", sub_line, re.I):
                        break
                    if sub_line.startswith("(") and sub_line.endswith(")"):
                        break
                    address_parts.append(sub_line)
                    # Often addresses end with 6 digit Indian PIN code
                    if re.search(r"\b\d{6}\b", sub_line):
                        break
                if address_parts:
                    result.fields["address"] = ", ".join(address_parts)
                    break

    # Hints for common missing fields
    if not result.fields.get("name"):
        result.hints.append("Enter full name as on ID proof.")
    if not result.fields.get("date_of_birth"):
        result.hints.append("Use DD/MM/YYYY for date of birth unless the form specifies otherwise.")
    if not result.fields.get("address"):
        result.hints.append("Provide complete address with PIN code.")
    return result
