#!/usr/bin/env python3
"""Build multilingual Indian government form line annotations for training line classifiers."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = ROOT / "data" / "annotations"
OUTPUT_FILE = ANNOTATIONS_DIR / "indian_form_lines.jsonl"

ANNOTATION_DATA = [
    # Document 1: UIDAI Aadhaar Form 1 (English)
    {
        "document_id": "uidai_aadhaar_form1_en",
        "source": "UIDAI Aadhaar Enrolment / Update Form 1 (English)",
        "lines": [
            ("Please fill in CAPITAL LETTERS only using blue or black ballpoint pen.", "instruction"),
            ("Do not staple or pin the photograph to the application form.", "instruction"),
            ("Tick the appropriate box corresponding to the demographic update required.", "instruction"),
            ("Residents must visit an authorized Aadhaar Seva Kendra for biometric capture.", "instruction"),
            ("Proof of Identity (POI) document containing Name and Photo", "required_document"),
            ("Proof of Address (POA) document containing Name and Address", "required_document"),
            ("Date of Birth (DOB) document containing Name and DOB", "required_document"),
            ("Valid Indian Passport or PAN Card as proof of identity", "required_document"),
            ("Full Name of the Resident as per proof document", "field"),
            ("Date of Birth (DD/MM/YYYY)", "field"),
            ("Gender (Male / Female / Transgender)", "field"),
            ("Care of (Father / Husband / Mother / Guardian)", "field"),
            ("House No., Building, Apartment name", "field"),
            ("Street, Road, Lane, Area, Landmark", "field"),
            ("Village / Town / City, Post Office", "field"),
            ("District, State, PIN Code", "field"),
            ("Toll-free Helpdesk: 1947", "contact_detail"),
            ("Email: help@uidai.gov.in", "contact_detail"),
            ("Website: https://resident.uidai.gov.in", "contact_detail"),
            ("I hereby declare that the information provided above is true, correct and complete.", "disclaimer"),
            ("Any false statement or forged document may attract penal consequences under the Aadhaar Act.", "disclaimer"),
            ("Government of India - Unique Identification Authority of India", "other"),
            ("Aadhaar Enrolment / Correction / Update Form (Field Version 2.0)", "other"),
            ("For Official Use Only - Verification Stamp and Signature", "other"),
        ],
    },
    # Document 2: UIDAI Aadhaar Handbook & Forms (Hindi)
    {
        "document_id": "uidai_aadhaar_handbook_hi",
        "source": "UIDAI Aadhaar Resident Handbook & Form (Hindi)",
        "lines": [
            ("कृपया फॉर्म को स्पष्ट एवं बड़े अक्षरों में भरें।", "instruction"),
            ("फॉर्म में किसी भी प्रकार की कटिंग या ओवरराइटिंग न करें।", "instruction"),
            ("केवल वैध एवं मूल दस्तावेजों की छायाप्रतियां ही संलग्न करें।", "instruction"),
            ("बायोमेट्रिक प्रमाणीकरण के लिए आधार सेवा केंद्र पर स्वयं उपस्थित होना अनिवार्य है।", "instruction"),
            ("पहचान का प्रमाण (POI) मूल दस्तावेज", "required_document"),
            ("निवास का प्रमाण (POA) सरकारी दस्तावेज", "required_document"),
            ("जन्म तिथि प्रमाण पत्र (नगर निगम अथवा सक्षम प्राधिकारी द्वारा जारी)", "required_document"),
            ("राशन कार्ड अथवा बिजली का बिल (पते के सत्यापन हेतु)", "required_document"),
            ("आवेदक का पूरा नाम", "field"),
            ("जन्म तिथि (दिन/माह/वर्ष)", "field"),
            ("लिंग (पुरुष / महिला / अन्य)", "field"),
            ("पिता या पति का नाम", "field"),
            ("स्थायी निवास का पता", "field"),
            ("पिन कोड एवं जिला", "field"),
            ("मोबाइल नंबर एवं ईमेल आईडी", "field"),
            ("आधार नंबर / नामांकन संख्या", "field"),
            ("टोल फ्री हेल्पलाइन नंबर: 1947", "contact_detail"),
            ("ईमेल संपर्क: help@uidai.gov.in", "contact_detail"),
            ("क्षेत्रीय कार्यालय: तीसरी मंजिल, साउथ विंग, खान मार्केट, नई दिल्ली", "contact_detail"),
            ("मैं प्रमाणित करता/करती हूँ कि ऊपर दी गई सभी जानकारियाँ मेरी जानकारी में सत्य हैं।", "disclaimer"),
            ("झूठी जानकारी प्रस्तुत करने पर आधार अधिनियम के अंतर्गत कानूनी कार्रवाई की जा सकती है।", "disclaimer"),
            ("भारत सरकार - भारतीय विशिष्ट पहचान प्राधिकरण", "other"),
            ("निवासी आधार नामांकन एवं अद्यतन प्रपत्र", "other"),
            ("केवल कार्यालयीन उपयोग हेतु", "other"),
        ],
    },
    # Document 3: UIDAI Aadhaar Handbook & Forms (Marathi)
    {
        "document_id": "uidai_aadhaar_handbook_mr",
        "source": "UIDAI Aadhaar Resident Handbook & Form (Marathi)",
        "lines": [
            ("कृपया अर्ज स्पष्ट अक्षरांमध्ये आणि काळ्या किंवा निळ्या शाईने भरावा.", "instruction"),
            ("छायाचित्रावर कोणतीही खूण किंवा स्टेपल पिन लावू नये.", "instruction"),
            ("पडताळणीसाठी अर्जासोबत अधिकृत कागदपत्रांच्या स्वसाक्षांकित प्रती जोडाव्यात.", "instruction"),
            ("बायोमेट्रिक नोंदीसाठी अर्जदाराने आधार केंद्रावर प्रत्यक्ष उपस्थित राहणे आवश्यक आहे.", "instruction"),
            ("ओळखीचा पुरावा (छायाचित्र असलेले अधिकृत ओळखपत्र)", "required_document"),
            ("रहिवासी पुरावा (लाईट बिल, पाणी बिल किंवा शिधापत्रिका)", "required_document"),
            ("जन्माचा दाखला (सक्षम प्राधिकार्‍याने दिलेला जन्म प्रमाणपत्र)", "required_document"),
            ("बँक पासबुक किंवा मतदार ओळखपत्र", "required_document"),
            ("अर्जदाराचे संपूर्ण नाव", "field"),
            ("जन्मतारीख (दिनांक/महिना/वर्ष)", "field"),
            ("लिंग (पुरुष / स्त्री / इतर)", "field"),
            ("पालकांचे / पतीचे नाव", "field"),
            ("कायमचा निवासी पत्ता", "field"),
            ("गाव, तालुका, जिल्हा व पिन कोड", "field"),
            ("मोबाइल क्रमांक व ईमेल पत्ता", "field"),
            ("आधार नोंदणी क्रमांक", "field"),
            ("ग्राहक सेवा केंद्र टोल फ्री क्रमांक: 1947", "contact_detail"),
            ("ईमेल पत्ता: help@uidai.gov.in", "contact_detail"),
            ("विभागीय कार्यालय: नरिमन पॉईंट, मुंबई ४०००२१", "contact_detail"),
            ("मी याद्वारे घोषित करतो/करते की वरील सर्व माहिती माझ्या माहितीनुसार खरी व बिनचूक आहे.", "disclaimer"),
            ("खोटी माहिती दिल्यास आधार कायद्यानुसार फौजदारी कारवाई केली जाऊ शकते.", "disclaimer"),
            ("भारत सरकार - भारतीय विशिष्ट ओळख प्राधिकरण", "other"),
            ("आधार नोंदणी व दुरुस्ती अर्ज", "other"),
            ("केवळ कार्यालयीन वापरासाठी", "other"),
        ],
    },
    # Document 4: Parivahan Driving Licence Form 2 (English)
    {
        "document_id": "parivahan_dl_form2_en",
        "source": "Ministry of Road Transport Driving Licence Form 2 (English)",
        "lines": [
            ("Submit the application along with self-attested medical certificate Form 1A.", "instruction"),
            ("Pay the prescribed fee online through the Parivahan Sarathi portal.", "instruction"),
            ("Book an appointment slot for the driving test at the relevant RTO.", "instruction"),
            ("Affix one passport size photograph and enclose two spare photographs.", "instruction"),
            ("Proof of Age (School Leaving Certificate or Birth Certificate)", "required_document"),
            ("Proof of Residence (Aadhaar Card, Passport, or Electricity Bill)", "required_document"),
            ("Medical Fitness Certificate Form 1A signed by a registered medical practitioner", "required_document"),
            ("Learner's Licence in original", "required_document"),
            ("Name of the Applicant in full", "field"),
            ("Father's / Husband's Name", "field"),
            ("Permanent Address with PIN Code", "field"),
            ("Present Address / Address for Communication", "field"),
            ("Blood Group and RH factor", "field"),
            ("Class of Vehicle applied for (MCWG / LMV / Transport)", "field"),
            ("Identification Marks (Mole or Scar)", "field"),
            ("Helpdesk Phone: 0120-2459169", "contact_detail"),
            ("Email: helpdesk-sarathi@gov.in", "contact_detail"),
            ("Portal: https://parivahan.gov.in", "contact_detail"),
            ("I declare that to the best of my knowledge and belief the particulars given above are true.", "disclaimer"),
            ("I am disqualified from holding or obtaining a driving licence for reasons stated.", "disclaimer"),
            ("Form 2 - Application for Grant of Learner's / Driving Licence", "other"),
            ("Rule 10 of the Central Motor Vehicles Rules, 1989", "other"),
            ("Office of the Regional Transport Officer", "other"),
        ],
    },
    # Document 5: Passport Seva Declaration Annexure A (English)
    {
        "document_id": "passport_annexure_a_en",
        "source": "Ministry of External Affairs Passport Annexure A (English)",
        "lines": [
            ("This declaration is to be executed on a non-judicial stamp paper of minimum value.", "instruction"),
            ("The declaration must be attested by a Judicial Magistrate or Notary Public.", "instruction"),
            ("Applicant must produce original verification documents during physical interview.", "instruction"),
            ("Standard Affidavit as per specimen Annexure A", "required_document"),
            ("Proof of Date of Birth and Nationality", "required_document"),
            ("Two recent colored photographs with white background", "required_document"),
            ("Name of the Applicant", "field"),
            ("Father's Name and Mother's Name", "field"),
            ("Date of Birth", "field"),
            ("Place of Birth", "field"),
            ("Passport Office Location", "field"),
            ("Passport Seva National Call Centre: 1800-258-1800", "contact_detail"),
            ("Official Portal: https://www.passportindia.gov.in", "contact_detail"),
            ("I solemnly affirm and declare that the statements made herein are true to my personal knowledge.", "disclaimer"),
            ("I am aware that making false statements in a passport application is an offence under the Passports Act 1967.", "disclaimer"),
            ("Government of India - Ministry of External Affairs", "other"),
            ("Annexure 'A' - Specimen Affidavit for a Passport", "other"),
        ],
    },
    # Document 6: Income Tax PAN Form 49A (Hindi)
    {
        "document_id": "income_tax_pan_49a_hi",
        "source": "Income Tax Department PAN Form 49A (Hindi)",
        "lines": [
            ("फॉर्म भरते समय केवल काले बॉलपॉइंट पेन का ही उपयोग करें।", "instruction"),
            ("फॉर्म पर दो पासपोर्ट आकार के रंगीन फोटो निर्धारित स्थान पर चिपकाएँ।", "instruction"),
            ("हस्ताक्षर केवल बॉक्स के अंदर ही करें, फोटो के ऊपर हस्ताक्षर न करें।", "instruction"),
            ("पहचान का प्रमाण पत्र (आधार कार्ड या मतदाता पहचान पत्र)", "required_document"),
            ("पते का प्रमाण पत्र (बैंक खाता विवरण या बिजली बिल)", "required_document"),
            ("जन्म तिथि का प्रमाण पत्र (दसवीं की अंकतालिका या जन्म प्रमाण पत्र)", "required_document"),
            ("पूरा नाम - उपनाम, प्रथम नाम, मध्य नाम", "field"),
            ("पिता का पूरा नाम", "field"),
            ("जन्म की तिथि (दिन/माह/वर्ष)", "field"),
            ("आवासीय पता एवं पिन कोड", "field"),
            ("पत्राचार हेतु टेलीफोन नंबर एवं ईमेल", "field"),
            ("आधार कार्ड संख्या (12 अंक)", "field"),
            ("आयकर संपर्क केंद्र टोल फ्री: 1800-180-1961", "contact_detail"),
            ("ईमेल: ask@incometax.gov.in", "contact_detail"),
            ("मैं एतद्द्वारा घोषित करता/करती हूँ कि इस प्रपत्र में दी गई जानकारी पूर्ण एवं सत्य है।", "disclaimer"),
            ("धारा 272बी के तहत गलत पैन विवरण देने पर ₹10,000 का जुर्माना लगाया जा सकता है।", "disclaimer"),
            ("भारत सरकार - आयकर विभाग", "other"),
            ("प्रपत्र संख्या 49क - स्थायी लेखा संख्या आवंटन हेतु आवेदन", "other"),
        ],
    },
    # Document 7: Maharashtra State Election Commission Voter Form 6 (Marathi)
    {
        "document_id": "voter_registration_form6_mr",
        "source": "State Election Commission Maharashtra Voter Form 6 (Marathi)",
        "lines": [
            ("नवीन मतदार नोंदणीसाठी वय १८ वर्षे पूर्ण असणे आवश्यक आहे.", "instruction"),
            ("अर्जासोबत चालू वर्षाचा पासपोर्ट आकाराचा रंगीत फोटो जोडावा.", "instruction"),
            ("सर्व कागदपत्रांच्या प्रती स्वच्छ व वाचनीय असाव्यात.", "instruction"),
            ("वयाचा पुरावा (शाळा सोडल्याचा दाखला किंवा जन्म प्रमाणपत्र)", "required_document"),
            ("रहिवासाचा पुरावा (लाईट बिल, रेशन कार्ड किंवा बँक पासबुक)", "required_document"),
            ("अर्जदाराचे संपूर्ण नाव (पहिले नाव, वडिलांचे/पतीचे नाव, आडनाव)", "field"),
            ("नातेवाईकाचे नाव व नाते", "field"),
            ("जन्मतारीख आणि वय", "field"),
            ("लिंग (पुरुष / स्त्री / तृतीयपंथी)", "field"),
            ("सध्याचा राहण्याचा पूर्ण पत्ता", "field"),
            ("विधानसभा मतदार संघाचे नाव व क्रमांक", "field"),
            ("मतदार हेल्पलाईन क्रमांक: 1950", "contact_detail"),
            ("संकेतस्थळ: https://voters.eci.gov.in", "contact_detail"),
            ("मी याद्वारे घोषित करतो/करते की मी भारताचा नागरिक असून या पत्त्यावर सामान्यतः रहिवासी आहे.", "disclaimer"),
            ("लोकप्रतिनिधित्व कायद्यानुसार खोटी माहिती देणे हा दंडनीय गुन्हा आहे.", "disclaimer"),
            ("भारत निवडणूक आयोग - मुख्य निवडणूक अधिकारी महाराष्ट्र", "other"),
            ("नमुना ६ - मतदार यादीत नाव समाविष्ट करण्यासाठी अर्ज", "other"),
        ],
    },
    # Document 8: GATE Examination Application Form (English)
    {
        "document_id": "gate_examination_form_en",
        "source": "IIT GATE Application Form (English)",
        "lines": [
            ("Candidates must carry original valid photo ID proof to the examination centre.", "instruction"),
            ("E-signature and thumb impression must be clearly uploaded without distortion.", "instruction"),
            ("Verify exam city preferences carefully before final submission.", "instruction"),
            ("Valid Photo ID (Passport / PAN Card / Voter ID / Aadhaar / College ID)", "required_document"),
            ("Degree Certificate or Certificate from Head of Department", "required_document"),
            ("Caste / PwD Certificate if claiming reservation benefits", "required_document"),
            ("Enrollment ID", "field"),
            ("Full Name of the Applicant", "field"),
            ("Date of Birth", "field"),
            ("Gender", "field"),
            ("Category (GEN / OBC-NCL / SC / ST / EWS)", "field"),
            ("Parent/Guardian's Name", "field"),
            ("Parent/Guardian's Mobile Number", "field"),
            ("Communication Address with PIN Code", "field"),
            ("Candidate Email ID", "field"),
            ("GATE Exam Paper Code", "field"),
            ("Zonal GATE Office Contact Phone: 022-25767068", "contact_detail"),
            ("Email: gateoffice@iitb.ac.in", "contact_detail"),
            ("Address: GATE Office, Powai, Mumbai 400076", "contact_detail"),
            ("I confirm that I fulfil the eligibility criteria to appear for the examination.", "disclaimer"),
            ("I understand that I may be liable for legal action for providing incorrect information.", "disclaimer"),
            ("Graduate Aptitude Test in Engineering - Application Confirmation", "other"),
        ],
    },
]


def main() -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in ANNOTATION_DATA:
        doc_id = doc["document_id"]
        source = doc["source"]
        for text, label in doc["lines"]:
            rows.append({
                "document_id": doc_id,
                "source": source,
                "text": text,
                "label": label,
            })

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(rows)} multilingual labeled lines across {len(ANNOTATION_DATA)} documents.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

