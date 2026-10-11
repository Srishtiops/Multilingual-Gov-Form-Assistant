#!/usr/bin/env python3
"""Expand the government-forms glossary and export glossary parallel training rows across en, hi, and mr."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXTRA_GOVERNMENT_TERMS = [
    {"en": "enrollment", "hi": "नामांकन", "mr": "नोंदणी"},
    {"en": "update request", "hi": "अद्यतन अनुरोध", "mr": "अद्यतन विनंती"},
    {"en": "resident", "hi": "निवासी", "mr": "रहिवासी"},
    {"en": "guardian", "hi": "अभिभावक", "mr": "पालक"},
    {"en": "declarant", "hi": "घोषणा कर्ता", "mr": "घोषणा करणारा"},
    {"en": "witness", "hi": "साक्षी", "mr": "साक्षीदार"},
    {"en": "photograph", "hi": "फोटोग्राफ", "mr": "छायाचित्र"},
    {"en": "passport size photograph", "hi": "पासपोर्ट आकार का फोटो", "mr": "पासपोर्ट आकाराचे छायाचित्र"},
    {"en": "affidavit", "hi": "शपथ पत्र", "mr": "प्रतिज्ञापत्र"},
    {"en": "notarized", "hi": "नोटरीकृत", "mr": "नोटरीकृत"},
    {"en": "declaration", "hi": "घोषणा", "mr": "घोषणा"},
    {"en": "undertaking", "hi": "वचनबद्धता", "mr": "हमीपत्र"},
    {"en": "mandatory field", "hi": "अनिवार्य क्षेत्र", "mr": "अनिवार्य माहिती"},
    {"en": "optional", "hi": "वैकल्पिक", "mr": "ऐच्छिक"},
    {"en": "gender", "hi": "लिंग", "mr": "लिंग"},
    {"en": "nationality", "hi": "राष्ट्रीयता", "mr": "राष्ट्रीयत्व"},
    {"en": "place of birth", "hi": "जन्म स्थान", "mr": "जन्मस्थान"},
    {"en": "father's name", "hi": "पिता का नाम", "mr": "वडिलांचे नाव"},
    {"en": "mother's name", "hi": "माता का नाम", "mr": "आईचे नाव"},
    {"en": "spouse name", "hi": "पति/पत्नी का नाम", "mr": "पती/पत्नीचे नाव"},
    {"en": "mobile number", "hi": "मोबाइल नंबर", "mr": "मोबाइल क्रमांक"},
    {"en": "email address", "hi": "ईमेल पता", "mr": "ईमेल पत्ता"},
    {"en": "pin code", "hi": "पिन कोड", "mr": "पिन कोड"},
    {"en": "state", "hi": "राज्य", "mr": "राज्य"},
    {"en": "district", "hi": "जिला", "mr": "जिल्हा"},
    {"en": "village", "hi": "गाँव", "mr": "गाव"},
    {"en": "taluka", "hi": "तहसील", "mr": "तालुका"},
    {"en": "police station", "hi": "थाना", "mr": "पोलीस ठाणे"},
    {"en": "voter id", "hi": "मतदाता पहचान पत्र", "mr": "मतदार ओळखपत्र"},
    {"en": "driving licence", "hi": "ड्राइविंग लाइसेंस", "mr": "वाहन चालविण्याचा परवाना"},
    {"en": "income tax", "hi": "आयकर", "mr": "आयकर"},
    {"en": "permanent account number", "hi": "स्थायी लेखा संख्या", "mr": "कायम खाते क्रमांक"},
    {"en": "acknowledgement", "hi": "पावती", "mr": "पोचपावती"},
    {"en": "office use only", "hi": "केवल कार्यालय हेतु", "mr": "फक्त कार्यालयासाठी"},
    {"en": "for official use", "hi": "आधिकारिक उपयोग हेतु", "mr": "अधिकृत वापरासाठी"},
    {"en": "instructions", "hi": "निर्देश", "mr": "सूचना"},
    {"en": "tick applicable box", "hi": "लागू बॉक्स पर निशान लगाएँ", "mr": "लागू चौकोनावर खूण करा"},
    {"en": "do not staple", "hi": "स्टेपल न करें", "mr": "स्टेपल करू नका"},
    {"en": "original documents", "hi": "मूल दस्तावेज़", "mr": "मूळ कागदपत्रे"},
    {"en": "photocopy", "hi": "फोटोकॉपी", "mr": "छायाप्रत"},
    {"en": "ration card", "hi": "राशन कार्ड", "mr": "शिधापत्रिका"},
    {"en": "caste certificate", "hi": "जाति प्रमाण पत्र", "mr": "जातीचा दाखला"},
    {"en": "domicile certificate", "hi": "मूल निवास प्रमाण पत्र", "mr": "अधिवास प्रमाणपत्र"},
    {"en": "income certificate", "hi": "आय प्रमाण पत्र", "mr": "उत्पन्नाचा दाखला"},
    {"en": "birth certificate", "hi": "जन्म प्रमाण पत्र", "mr": "जन्म दाखला"},
    {"en": "death certificate", "hi": "मृत्यु प्रमाण पत्र", "mr": "मृत्यू दाखला"},
    {"en": "marriage certificate", "hi": "विवाह प्रमाण पत्र", "mr": "विवाह नोंदणी प्रमाणपत्र"},
    {"en": "non creamy layer", "hi": "गैर क्रीमी लेयर प्रमाण पत्र", "mr": "नॉन क्रिमिलेअर दाखला"},
    {"en": "gazetted officer", "hi": "राजपत्रित अधिकारी", "mr": "राजपत्रित अधिकारी"},
    {"en": "self attestation", "hi": "स्वप्रमाणन", "mr": "स्वसाक्षांकन"},
    {"en": "self attested copy", "hi": "स्वप्रमाणित प्रति", "mr": "स्वसाक्षांकित प्रत"},
    {"en": "thumb impression", "hi": "अंगूठे का निशान", "mr": "अंगठ्याचा ठसा"},
    {"en": "signature of applicant", "hi": "आवेदक के हस्ताक्षर", "mr": "अर्जदाराची स्वाक्षरी"},
    {"en": "date of issue", "hi": "जारी करने की तिथि", "mr": "जारी केल्याची तारीख"},
    {"en": "issuing authority", "hi": "जारीकर्ता प्राधिकारी", "mr": "सक्षम प्राधिकारी"},
    {"en": "validity period", "hi": "वैधता अवधि", "mr": "वैधतेचा कालावधी"},
    {"en": "educational qualification", "hi": "शैक्षणिक योग्यता", "mr": "शैक्षणिक पात्रता"},
    {"en": "examination centre", "hi": "परीक्षा केंद्र", "mr": "परीक्षा केंद्र"},
    {"en": "admit card", "hi": "प्रवेश पत्र", "mr": "प्रवेशपत्र"},
    {"en": "hall ticket", "hi": "हॉल टिकट", "mr": "हॉल तिकीट"},
    {"en": "bank passbook", "hi": "बैंक पासबुक", "mr": "बँक पासबुक"},
    {"en": "account number", "hi": "खाता संख्या", "mr": "खाते क्रमांक"},
    {"en": "ifsc code", "hi": "आईएफएससी कोड", "mr": "आयएफएससी कोड"},
    {"en": "branch name", "hi": "शाखा का नाम", "mr": "शाखेचे नाव"},
    {"en": "challan number", "hi": "चालान संख्या", "mr": "चलन क्रमांक"},
    {"en": "application status", "hi": "आवेदन की स्थिति", "mr": "अर्जाची स्थिती"},
    {"en": "grievance redressal", "hi": "शिकायत निवारण", "mr": "तक्रार निवारण"},
    {"en": "revenue department", "hi": "राजस्व विभाग", "mr": "महसूल विभाग"},
    {"en": "municipal corporation", "hi": "नगर निगम", "mr": "महानगरपालिका"},
    {"en": "gram panchayat", "hi": "ग्राम पंचायत", "mr": "ग्रामपंचायत"},
    {"en": "electricity bill", "hi": "बिजली का बिल", "mr": "वीज बिल"},
    {"en": "water bill", "hi": "पानी का बिल", "mr": "पाणी बिल"},
    {"en": "rent agreement", "hi": "किरायानामा", "mr": "भाडे करार"},
    {"en": "property tax receipt", "hi": "संपत्ति कर रसीद", "mr": "मालमत्ता कर पावती"},
    {"en": "verified and found correct", "hi": "सत्यापित और सही पाया गया", "mr": "तपासले व बरोबर आढळले"},
    {"en": "scrutiny of application", "hi": "आवेदन की जांच", "mr": "अर्जाची छाननी"},
    {"en": "provisional certificate", "hi": "औपबंधिक प्रमाण पत्र", "mr": "तात्पुरते प्रमाणपत्र"},
    {"en": "transfer certificate", "hi": "स्थानांतरण प्रमाण पत्र", "mr": "शाळा सोडल्याचा दाखला"},
    {"en": "caste validity certificate", "hi": "जाति वैधता प्रमाण पत्र", "mr": "जात पडताळणी प्रमाणपत्र"},
    {"en": "differently abled", "hi": "दिव्यांग", "mr": "दिव्यांग"},
    {"en": "minor applicant", "hi": "नाबालिग आवेदक", "mr": "अज्ञान अर्जदार"},
    {"en": "marital status", "hi": "वैवाहिक स्थिति", "mr": "वैवाहिक स्थिती"},
    {"en": "married", "hi": "विवाहित", "mr": "विवाहित"},
    {"en": "unmarried", "hi": "अविवाहित", "mr": "अविवाहित"},
    {"en": "widow", "hi": "विधवा", "mr": "विधवा"},
    {"en": "divorced", "hi": "तलाकशुदा", "mr": "घटस्फोटित"},
    {"en": "occupation", "hi": "व्यवसाय", "mr": "व्यवसाय"},
    {"en": "annual income", "hi": "वार्षिक आय", "mr": "वार्षिक उत्पन्न"},
    {"en": "source of income", "hi": "आय का स्रोत", "mr": "उत्पन्नाचा स्रोत"},
    {"en": "government employee", "hi": "सरकारी कर्मचारी", "mr": "शासकीय कर्मचारी"},
    {"en": "pensioner", "hi": "पेंशनभोगी", "mr": "निवृत्तीवेतनधारक"},
    {"en": "sub registrar office", "hi": "उप पंजीयक कार्यालय", "mr": "दुय्यम निबंधक कार्यालय"},
    {"en": "collector office", "hi": "कलेक्ट्रेट कार्यालय", "mr": "जिल्हाधिकारी कार्यालय"},
    {"en": "tehsildar office", "hi": "तहसीलदार कार्यालय", "mr": "तहसीलदार कार्यालय"},
    {"en": "stamp duty", "hi": "स्टाम्प शुल्क", "mr": "मुद्रांक शुल्क"},
    {"en": "registration fee", "hi": "पंजीकरण शुल्क", "mr": "नोंदणी शुल्क"},
    {"en": "non refundable", "hi": "अप्रतिदेय", "mr": "परतावा न मिळणारे"},
    {"en": "submission deadline", "hi": "जमा करने की अंतिम तिथि", "mr": "सादर करण्याची अंतिम तारीख"},
    {"en": "incomplete applications will be rejected", "hi": "अपूर्ण आवेदन निरस्त कर दिए जाएंगे", "mr": "अपूर्ण अर्ज फेटाळले जातील"},
    {"en": "attach all required documents", "hi": "सभी आवश्यक दस्तावेज संलग्न करें", "mr": "सर्व आवश्यक कागदपत्रे जोडा"},
    {"en": "bring original documents for verification", "hi": "सत्यापन हेतु मूल दस्तावेज साथ लाएं", "mr": "पडताळणीसाठी मूळ कागदपत्रे सोबत आणा"},
]


def merge_terms(existing: list[dict], extra: list[dict]) -> list[dict]:
    by_en: dict[str, dict] = {}
    for term in existing + extra:
        key = re.sub(r"\s+", " ", (term.get("en") or "").strip().lower())
        if not key:
            continue
        by_en[key] = {**by_en.get(key, {}), **term}
    return sorted(by_en.values(), key=lambda item: item.get("en", "").lower())


def export_parallel(terms: list[dict], out_path: Path) -> int:
    rows: list[dict] = []
    for term in terms:
        en = (term.get("en") or "").strip()
        hi = (term.get("hi") or "").strip()
        mr = (term.get("mr") or "").strip()

        if en and hi:
            rows.append({"src_lang": "en", "tgt_lang": "hi", "src": en, "tgt": hi, "source": "glossary"})
            rows.append({"src_lang": "hi", "tgt_lang": "en", "src": hi, "tgt": en, "source": "glossary"})
        if en and mr:
            rows.append({"src_lang": "en", "tgt_lang": "mr", "src": en, "tgt": mr, "source": "glossary"})
            rows.append({"src_lang": "mr", "tgt_lang": "en", "src": mr, "tgt": en, "source": "glossary"})
        if hi and mr:
            rows.append({"src_lang": "hi", "tgt_lang": "mr", "src": hi, "tgt": mr, "source": "glossary"})
            rows.append({"src_lang": "mr", "tgt_lang": "hi", "src": mr, "tgt": hi, "source": "glossary"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    root = args.config.parent.resolve()
    glossary_path = root / "data" / "glossary" / "government_forms.json"
    with glossary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    merged = merge_terms(payload.get("terms", []), EXTRA_GOVERNMENT_TERMS)
    payload["terms"] = merged
    payload["term_count"] = len(merged)
    payload["status"] = "reviewed_indian_government_forms"
    with glossary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    parallel_path = root / "data" / "training" / "parallel" / "glossary_en_hi_mr.jsonl"
    count = export_parallel(merged, parallel_path)
    print(f"Glossary: {len(merged)} terms -> {glossary_path}")
    print(f"Parallel rows across all language pairs: {count} -> {parallel_path}")


if __name__ == "__main__":
    main()
