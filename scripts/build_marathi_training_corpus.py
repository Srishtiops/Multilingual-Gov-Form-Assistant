#!/usr/bin/env python3
"""Build high-quality Marathi parallel training data for Marathi-English and Hindi-Marathi translation."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARALLEL_DIR = ROOT / "data" / "training" / "parallel"
RAW_DIR = ROOT / "data" / "raw" / "samanantar"
GLOSSARY_PATH = ROOT / "data" / "glossary" / "government_forms.json"
ANNOTATIONS_PATH = ROOT / "data" / "annotations" / "indian_form_lines.jsonl"


# Authentic trilingual government sentences (Maharashtra Aaple Sarkar, UIDAI, RTO, Revenue & Education)
MARATHI_GOV_SENTENCES = [
    {
        "en": "Please fill the application form carefully in capital letters.",
        "hi": "कृपया आवेदन पत्र को ध्यानपूर्वक बड़े अक्षरों में भरें।",
        "mr": "कृपया अर्ज काळजीपूर्वक मोठ्या अक्षरांमध्ये भरावा.",
    },
    {
        "en": "Attach self-attested photocopies of all required documents.",
        "hi": "सभी आवश्यक दस्तावेजों की स्वप्रमाणित छायाप्रतियां संलग्न करें।",
        "mr": "सर्व आवश्यक कागदपत्रांच्या स्वसाक्षांकित छायाप्रती जोडाव्यात.",
    },
    {
        "en": "Applicant must produce the original documents at the time of verification.",
        "hi": "सत्यापन के समय आवेदक को मूल दस्तावेज प्रस्तुत करने होंगे।",
        "mr": "पडताळणीच्या वेळी अर्जदाराने मूळ कागदपत्रे सादर करणे आवश्यक आहे.",
    },
    {
        "en": "Proof of Identity: Voter ID Card, PAN Card, or Passport.",
        "hi": "पहचान का प्रमाण: मतदाता पहचान पत्र, पैन कार्ड, या पासपोर्ट।",
        "mr": "ओळखीचा पुरावा: मतदार ओळखपत्र, पॅन कार्ड, किंवा पासपोर्ट.",
    },
    {
        "en": "Proof of Residence: Ration Card, Electricity Bill, or Domicile Certificate.",
        "hi": "निवास का प्रमाण: राशन कार्ड, बिजली बिल, या मूल निवास प्रमाण पत्र।",
        "mr": "रहिवासी पुरावा: शिधापत्रिका, वीज बिल, किंवा अधिवास प्रमाणपत्र.",
    },
    {
        "en": "Income Certificate issued by the competent Tehsildar or Sub-Divisional Officer.",
        "hi": "सक्षम तहसीलदार या अनुविभागीय अधिकारी द्वारा जारी आय प्रमाण पत्र।",
        "mr": "सक्षम तहसीलदार किंवा उपविभागीय अधिकाऱ्यांनी दिलेला उत्पन्नाचा दाखला.",
    },
    {
        "en": "Caste Certificate and Non-Creamy Layer Certificate are mandatory for reservation.",
        "hi": "आरक्षण के लिए जाति प्रमाण पत्र और गैर-क्रीमी लेयर प्रमाण पत्र अनिवार्य हैं।",
        "mr": "आरक्षणासाठी जातीचा दाखला आणि नॉन-क्रिमिलेअर प्रमाणपत्र अनिवार्य आहे.",
    },
    {
        "en": "Birth Certificate issued by Municipal Corporation or Gram Panchayat.",
        "hi": "नगर निगम या ग्राम पंचायत द्वारा जारी जन्म प्रमाण पत्र।",
        "mr": "महानगरपालिका किंवा ग्रामपंचायतीने दिलेला जन्म दाखला.",
    },
    {
        "en": "Application for grant of Learner's Licence or Driving Licence.",
        "hi": "लर्नर लाइसेंस या ड्राइविंग लाइसेंस जारी करने हेतु आवेदन।",
        "mr": "शिकाऊ परवाना किंवा वाहन चालविण्याचा परवाना मिळण्यासाठी अर्ज.",
    },
    {
        "en": "Fee can be paid online through Net Banking, Debit Card, or UPI.",
        "hi": "शुल्क का भुगतान नेट बैंकिंग, डेबिट कार्ड या यूपीआई के माध्यम से ऑनलाइन किया जा सकता है।",
        "mr": "शुल्क नेट बँकिंग, डेबिट कार्ड किंवा यूपीआय द्वारे ऑनलाइन भरता येईल.",
    },
    {
        "en": "I hereby declare that all information furnished in this application is true and correct.",
        "hi": "मैं एतद्द्वारा घोषणा करता/करती हूँ कि इस आवेदन में दी गई सभी जानकारी सत्य और सही है।",
        "mr": "मी याद्वारे घोषित करतो/करते की या अर्जात दिलेली सर्व माहिती खरी व बिनचूक आहे.",
    },
    {
        "en": "Any false declaration or forged document shall lead to criminal prosecution.",
        "hi": "कोई भी गलत घोषणा या जाली दस्तावेज आपराधिक अभियोजन का कारण बनेगा।",
        "mr": "कोणतीही खोटी माहिती किंवा बनावट कागदपत्र आढळल्यास फौजदारी गुन्हा दाखल केला जाईल.",
    },
    {
        "en": "Permanent Account Number (PAN) Card application Form 49A.",
        "hi": "स्थायी खाता संख्या (पैन) कार्ड आवेदन प्रपत्र 49A।",
        "mr": "कायम खाते क्रमांक (पॅन) कार्ड अर्ज फॉर्म ४९ए.",
    },
    {
        "en": "Download the acknowledgement receipt after successful submission.",
        "hi": "सफल प्रस्तुतीकरण के बाद पावती रसीद डाउनलोड करें।",
        "mr": "अर्ज यशस्वीरीत्या सादर केल्यानंतर पोचपावती डाउनलोड करा.",
    },
    {
        "en": "Applicant's Full Name as per Aadhaar Card.",
        "hi": "आधार कार्ड के अनुसार आवेदक का पूरा नाम।",
        "mr": "आधार कार्डनुसार अर्जदाराचे संपूर्ण नाव.",
    },
    {
        "en": "Date of Birth (Day / Month / Year).",
        "hi": "जन्म तिथि (दिन / माह / वर्ष)।",
        "mr": "जन्मतारीख (दिवस / महिना / वर्ष).",
    },
    {
        "en": "Gender: Male / Female / Transgender.",
        "hi": "लिंग: पुरुष / महिला / तृतीयपंथी।",
        "mr": "लिंग: पुरुष / स्त्री / तृतीयपंथी.",
    },
    {
        "en": "Father's Name / Husband's Name / Guardian's Name.",
        "hi": "पिता का नाम / पति का नाम / अभिभावक का नाम।",
        "mr": "वडिलांचे नाव / पतीचे नाव / पालकांचे नाव.",
    },
    {
        "en": "Mother's Name.",
        "hi": "माता का नाम।",
        "mr": "आईचे नाव.",
    },
    {
        "en": "Permanent Residential Address.",
        "hi": "स्थायी आवासीय पता।",
        "mr": "कायमचा रहिवासी पत्ता.",
    },
    {
        "en": "Address for Correspondence / Communication.",
        "hi": "पत्राचार का पता।",
        "mr": "पत्रव्यवहाराचा पत्ता.",
    },
    {
        "en": "Mobile Number and Email Address.",
        "hi": "मोबाइल नंबर और ईमेल पता।",
        "mr": "मोबाइल क्रमांक आणि ईमेल पत्ता.",
    },
    {
        "en": "Category: General / Scheduled Caste / Scheduled Tribe / Other Backward Class.",
        "hi": "वर्ग: सामान्य / अनुसूचित जाति / अनुसूचित जनजाति / अन्य पिछड़ा वर्ग।",
        "mr": "प्रवर्ग: खुला / अनुसूचित जाती / अनुसूचित जमाती / इतर मागासवर्गीय.",
    },
    {
        "en": "Signature of the Applicant / Thumb Impression.",
        "hi": "आवेदक के हस्ताक्षर / अंगूठे का निशान।",
        "mr": "अर्जदाराची स्वाक्षरी / अंगठ्याचा ठसा.",
    },
    {
        "en": "Passport size recent colour photograph.",
        "hi": "पासपोर्ट आकार का हालिया रंगीन फोटो।",
        "mr": "पासपोर्ट आकाराचे अलिकडील रंगीत छायाचित्र.",
    },
    {
        "en": "Land Records Extract (7/12 Extract) issued by Talathi.",
        "hi": "तलाठी द्वारा जारी भूमि अभिलेख उद्धरण (7/12 उद्धरण)।",
        "mr": "तलाठ्यांनी दिलेला सातबारा उतारा (७/१२ उतारा).",
    },
    {
        "en": "Death Certificate issued by competent health authority.",
        "hi": "सक्षम स्वास्थ्य प्राधिकारी द्वारा जारी मृत्यु प्रमाण पत्र।",
        "mr": "सक्षम आरोग्य प्राधिकरणाने दिलेला मृत्यू दाखला.",
    },
    {
        "en": "Marriage Registration Certificate.",
        "hi": "विवाह पंजीकरण प्रमाण पत्र।",
        "mr": "विवाह नोंदणी प्रमाणपत्र.",
    },
    {
        "en": "Electricity consumer number and billing address.",
        "hi": "बिजली उपभोक्ता संख्या और बिलिंग पता।",
        "mr": "वीज ग्राहक क्रमांक आणि बिलिंग पत्ता.",
    },
    {
        "en": "Government of Maharashtra - Public Services Portal.",
        "hi": "महाराष्ट्र सरकार - लोक सेवा पोर्टल।",
        "mr": "महाराष्ट्र शासन - आपले सरकार लोकसेवा पोर्टल.",
    },
]


def expand_samanantar_mr():
    """Extract up to 2500 high-quality rows from mr_en_sample.jsonl for training."""
    raw_mr = RAW_DIR / "mr_en_sample.jsonl"
    if not raw_mr.exists():
        print(f"Skipping Samanantar MR: {raw_mr} not found")
        return []
    
    pairs = []
    with raw_mr.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            en = (item.get("src") or "").strip()
            mr = (item.get("tgt") or "").strip()
            # Basic sanity filters
            if not en or not mr:
                continue
            if len(en) < 10 or len(mr) < 8:
                continue
            if "[" in mr and "]" in mr and len(mr) < 20:
                continue
            pairs.append({"src_lang": "en", "tgt_lang": "mr", "src": en, "tgt": mr})
            pairs.append({"src_lang": "mr", "tgt_lang": "en", "src": mr, "tgt": en})
            if len(pairs) >= 5000:
                break
    return pairs


def build_aligned_annotation_pairs():
    """Build trilingual pairs from the 1-to-1 UIDAI Aadhaar annotations."""
    if not ANNOTATIONS_PATH.exists():
        return []
    
    rows = []
    with ANNOTATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    en_rows = [r for r in rows if r.get("document_id") == "uidai_aadhaar_form1_en"]
    hi_rows = [r for r in rows if r.get("document_id") == "uidai_aadhaar_handbook_hi"]
    mr_rows = [r for r in rows if r.get("document_id") == "uidai_aadhaar_handbook_mr"]
    
    pairs = []
    min_len = min(len(en_rows), len(hi_rows), len(mr_rows))
    for i in range(min_len):
        en_t = en_rows[i]["text"].strip()
        hi_t = hi_rows[i]["text"].strip()
        mr_t = mr_rows[i]["text"].strip()
        
        # All 6 directions
        pairs.extend([
            {"src_lang": "en", "tgt_lang": "mr", "src": en_t, "tgt": mr_t},
            {"src_lang": "mr", "tgt_lang": "en", "src": mr_t, "tgt": en_t},
            {"src_lang": "hi", "tgt_lang": "mr", "src": hi_t, "tgt": mr_t},
            {"src_lang": "mr", "tgt_lang": "hi", "src": mr_t, "tgt": hi_t},
            {"src_lang": "en", "tgt_lang": "hi", "src": en_t, "tgt": hi_t},
            {"src_lang": "hi", "tgt_lang": "en", "src": hi_t, "tgt": en_t},
        ])
    return pairs


def build_gov_sentence_pairs():
    pairs = []
    for item in MARATHI_GOV_SENTENCES:
        en = item["en"]
        hi = item["hi"]
        mr = item["mr"]
        pairs.extend([
            {"src_lang": "hi", "tgt_lang": "mr", "src": hi, "tgt": mr},
            {"src_lang": "mr", "tgt_lang": "hi", "src": mr, "tgt": hi},
            {"src_lang": "en", "tgt_lang": "mr", "src": en, "tgt": mr},
            {"src_lang": "mr", "tgt_lang": "en", "src": mr, "tgt": en},
            {"src_lang": "en", "tgt_lang": "hi", "src": en, "tgt": hi},
            {"src_lang": "hi", "tgt_lang": "en", "src": hi, "tgt": en},
        ])
    return pairs


def build_glossary_pairs():
    if not GLOSSARY_PATH.exists():
        return []
    with GLOSSARY_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    terms = payload.get("terms", [])
    pairs = []
    for t in terms:
        en = (t.get("en") or "").strip()
        hi = (t.get("hi") or "").strip()
        mr = (t.get("mr") or "").strip()
        if en and mr:
            pairs.append({"src_lang": "en", "tgt_lang": "mr", "src": en, "tgt": mr})
            pairs.append({"src_lang": "mr", "tgt_lang": "en", "src": mr, "tgt": en})
        if hi and mr:
            pairs.append({"src_lang": "hi", "tgt_lang": "mr", "src": hi, "tgt": mr})
            pairs.append({"src_lang": "mr", "tgt_lang": "hi", "src": mr, "tgt": hi})
        if en and hi:
            pairs.append({"src_lang": "en", "tgt_lang": "hi", "src": en, "tgt": hi})
            pairs.append({"src_lang": "hi", "tgt_lang": "en", "src": hi, "tgt": en})
    return pairs


def main():
    PARALLEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Government sentences & forms
    gov_pairs = build_gov_sentence_pairs()
    anno_pairs = build_aligned_annotation_pairs()
    gloss_pairs = build_glossary_pairs()
    
    direct_mr_pairs = gov_pairs + anno_pairs + gloss_pairs
    out_gov_mr = PARALLEL_DIR / "marathi_gov_direct.jsonl"
    with out_gov_mr.open("w", encoding="utf-8") as f:
        for p in direct_mr_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(direct_mr_pairs)} direct Marathi/Hindi/English gov pairs -> {out_gov_mr}")
    
    # 2. Expanded Samanantar Marathi-English
    samanantar_mr_pairs = expand_samanantar_mr()
    out_samanantar_mr = PARALLEL_DIR / "samanantar_mr.jsonl"
    with out_samanantar_mr.open("w", encoding="utf-8") as f:
        for p in samanantar_mr_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(samanantar_mr_pairs)} Samanantar MR pairs -> {out_samanantar_mr}")


if __name__ == "__main__":
    main()

