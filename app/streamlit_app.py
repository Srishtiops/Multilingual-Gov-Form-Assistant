from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gov_form_assistant.pipeline import GovFormPipeline  # noqa: E402
from gov_form_assistant.annotation_data import (  # noqa: E402
    INDIAN_LINE_LABELS,
    document_key,
    extract_annotation_lines,
    save_document_annotations,
)

CONFIG = ROOT / "config.yaml"


@st.cache_resource
def get_pipeline(use_gpu: bool):
    return GovFormPipeline(CONFIG, use_gpu=use_gpu, lazy=True)


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Multilingual Indian Gov Form Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, professional government portal UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    /* Adaptive theme support: Crystal clear contrast in both Dark and Light mode */
    .field-card {
        background-color: rgba(148, 163, 184, 0.08) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
    }
    .field-title {
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    .field-value {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    .badge {
        display: inline-block !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        margin-right: 6px !important;
    }
    .badge-primary { background-color: rgba(99, 102, 241, 0.2) !important; color: #818cf8 !important; }
    .badge-success { background-color: rgba(34, 197, 94, 0.2) !important; color: #4ade80 !important; }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2) !important; color: #fbbf24 !important; }
    .badge-info { background-color: rgba(14, 165, 233, 0.2) !important; color: #38bdf8 !important; }
    .instruction-item {
        background-color: rgba(59, 130, 246, 0.12) !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        border-radius: 0 8px 8px 0 !important;
        font-size: 0.98rem !important;
        line-height: 1.5 !important;
    }
    .document-item {
        background-color: rgba(34, 197, 94, 0.12) !important;
        border-left: 4px solid #22c55e !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        border-radius: 0 8px 8px 0 !important;
        font-size: 0.98rem !important;
        line-height: 1.5 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown('<div class="main-header">🏛️ Multilingual Indian Government Form Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered OCR, Trilingual Translation (English / Hindi / Marathi), Structured Field Extraction & Grounded Q&A</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Assistant Settings")
use_gpu = st.sidebar.checkbox("Accelerate with GPU (CUDA)", value=torch.cuda.is_available())

source_lang = st.sidebar.selectbox(
    "Document Source Language",
    ["auto", "en", "hi", "mr"],
    format_func=lambda code: {
        "auto": "🌐 Auto-Detect Language",
        "en": "🇬🇧 English",
        "hi": "🇮🇳 Hindi (हिंदी)",
        "mr": "🚩 Marathi (मराठी)",
    }[code],
)

target_lang = st.sidebar.selectbox(
    "Display & Translation Language",
    ["en", "hi", "mr"],
    format_func=lambda x: {
        "en": "🇬🇧 English",
        "hi": "🇮🇳 Hindi (हिंदी)",
        "mr": "🚩 Marathi (मराठी)",
    }[x],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Active AI Models:**
    - 🔍 **OCR**: EasyOCR (En / Hi / Mr)
    - 🌐 **Translation**: NLLB-200 Fine-Tuned (Gov Domain)
    - 💬 **Q&A**: Flan-T5 + Grounded Rule Engine
    - 🏷️ **Line Classifier**: Trained Indian Form Models
    """
)

# --- File Uploader ---
upload = st.file_uploader(
    "Upload Official Government Form (PDF or Image)",
    type=["pdf", "png", "jpg", "jpeg", "webp"],
    help="Supports Aadhaar, Driving Licence, PAN, Passport, Ration Card, GATE, and state service forms in English, Hindi, or Marathi.",
)

if upload:
    upload_bytes = upload.getvalue()
    upload_hash = sha256(upload_bytes).hexdigest()
    result_key = (upload_hash, source_lang, target_lang, use_gpu)

    if st.session_state.get("result_key") != result_key:
        tmp = ROOT / "data" / "uploads" / f"{upload_hash}{Path(upload.name).suffix.lower()}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(upload_bytes)
        with st.spinner("Processing document: Performing OCR, field parsing, and multilingual translation..."):
            pipe = get_pipeline(use_gpu)
            st.session_state["result"] = pipe.process_file(
                tmp,
                target_lang=target_lang,
                source_lang=source_lang,
            )
            st.session_state["result_key"] = result_key
            st.session_state.pop("qa_answer", None)
            st.session_state.pop("qa_result", None)

    result = st.session_state["result"]
    pipe = get_pipeline(use_gpu)

    # --- Top KPI Summary Cards ---
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Detected Lines", len([line for line in result.ocr_text.splitlines() if line.strip()]))
    with col_kpi2:
        conf_str = f"{result.ocr_confidence:.0%}" if result.ocr_confidence is not None else "Native PDF"
        st.metric("OCR Quality / Confidence", conf_str)
    with col_kpi3:
        st.metric("Extracted Key Fields", len(result.fields.fields))
    with col_kpi4:
        st.metric("Required Documents", len(result.analysis.required_documents))

    if result.ocr_lines:
        weak_lines = [line for line in result.ocr_lines if float(line["confidence"]) < 0.55]
        if weak_lines:
            with st.expander(f"⚠️ Review {len(weak_lines)} low-confidence OCR lines"):
                st.dataframe(weak_lines, hide_index=True, use_container_width=True)

    # --- Main Navigation Tabs ---
    tab_fields, tab_guidance, tab_qa, tab_translation, tab_nlp = st.tabs(
        [
            "📋 Extracted Form Fields",
            "✅ Document Checklist & Instructions",
            "💬 Ask Form Assistant (Q&A)",
            "🌐 Document Translation",
            "🔬 NLP & Text Analysis",
        ]
    )

    # ==========================================
    # TAB 1: EXTRACTED FORM FIELDS
    # ==========================================
    with tab_fields:
        st.subheader("Citizen Details Extracted from Document")
        st.caption("Automatically detected and normalized across English, Hindi, and Marathi patterns.")

        fields = result.fields.fields
        if not fields:
            st.info("No pre-filled fields were detected in this document. It may be a blank form or general guidelines.")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### 👤 Personal Information")

                # Name
                name_val = fields.get("name") or fields.get("applicant_name")
                if name_val:
                    st.markdown(
                        f'<div class="field-card"><div class="field-title">Applicant Full Name</div><div class="field-value">{name_val}</div></div>',
                        unsafe_allow_html=True,
                    )

                # Parent / Guardian Name
                parent_val = fields.get("parent_name") or fields.get("father_name")
                if parent_val:
                    st.markdown(
                        f'<div class="field-card"><div class="field-title">Parent / Guardian / Spouse Name</div><div class="field-value">{parent_val}</div></div>',
                        unsafe_allow_html=True,
                    )

                # Date of Birth & Gender
                sub_c1, sub_c2 = st.columns(2)
                dob_val = fields.get("date_of_birth") or fields.get("dob")
                gender_val = fields.get("gender")

                with sub_c1:
                    if dob_val:
                        st.markdown(
                            f'<div class="field-card"><div class="field-title">Date of Birth</div><div class="field-value">{dob_val}</div></div>',
                            unsafe_allow_html=True,
                        )
                with sub_c2:
                    if gender_val:
                        st.markdown(
                            f'<div class="field-card"><div class="field-title">Gender</div><div class="field-value">{gender_val}</div></div>',
                            unsafe_allow_html=True,
                        )

                # Category
                cat_val = fields.get("category")
                if cat_val:
                    st.markdown(
                        f'<div class="field-card"><div class="field-title">Category / Reservation</div><div class="field-value">{cat_val}</div></div>',
                        unsafe_allow_html=True,
                    )

            with col_right:
                st.markdown("#### 🆔 Contact & Identification")

                # ID Number / Enrollment
                id_val = fields.get("id_number")
                if id_val:
                    st.markdown(
                        f'<div class="field-card"><div class="field-title">Document ID / Enrollment No.</div><div class="field-value">{id_val}</div></div>',
                        unsafe_allow_html=True,
                    )

                # Phone & Email
                sub_c3, sub_c4 = st.columns(2)
                phone_val = fields.get("phone") or fields.get("mobile")
                email_val = fields.get("email")

                with sub_c3:
                    if phone_val:
                        st.markdown(
                            f'<div class="field-card"><div class="field-title">Contact Phone</div><div class="field-value">{phone_val}</div></div>',
                            unsafe_allow_html=True,
                        )
                with sub_c4:
                    if email_val:
                        st.markdown(
                            f'<div class="field-card"><div class="field-title">Email ID</div><div class="field-value">{email_val}</div></div>',
                            unsafe_allow_html=True,
                        )

                # Address
                addr_val = fields.get("address")
                if addr_val:
                    st.markdown(
                        f'<div class="field-card"><div class="field-title">Residential / Permanent Address</div><div class="field-value">{addr_val}</div></div>',
                        unsafe_allow_html=True,
                    )

            # Extra detected fields
            standard_keys = {"name", "applicant_name", "parent_name", "father_name", "date_of_birth", "dob", "gender", "category", "id_number", "phone", "mobile", "email", "address"}
            other_fields = {k: v for k, v in fields.items() if k not in standard_keys}
            if other_fields:
                st.markdown("#### Other Detected Fields")
                st.json(other_fields)

        if result.fields.hints:
            st.markdown("---")
            st.markdown("##### 💡 Form Completion Hints")
            for hint in result.fields.hints:
                st.caption(f"• {hint}")

    # ==========================================
    # TAB 2: INSTRUCTIONS & CHECKLIST
    # ==========================================
    with tab_guidance:
        col_inst, col_docs = st.columns(2)

        with col_inst:
            st.subheader("📌 Key Instructions")
            if result.analysis.instructions:
                for item in result.analysis.instructions:
                    st.markdown(f'<div class="instruction-item">{item}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific operational instructions were detected in this document.")

        with col_docs:
            st.subheader("📑 Required Supporting Documents")
            if result.analysis.required_documents:
                for doc in result.analysis.required_documents:
                    st.markdown(f'<div class="document-item">✓ {doc}</div>', unsafe_allow_html=True)
            else:
                st.info("No explicit supporting documents were mentioned in the text.")

        st.markdown("---")
        st.subheader(f"💡 Plain Language Citizen Summary ({target_lang.upper()})")
        if result.simplified:
            st.info(result.simplified)
        else:
            st.caption("No instructions were available for simplification.")

    # ==========================================
    # TAB 3: Q&A ASSISTANT
    # ==========================================
    with tab_qa:
        st.subheader("Ask Questions About This Form")
        st.caption("Answers are verified and grounded directly in the document text to eliminate hallucinations.")

        # Quick question recommendation pills
        st.markdown("**Quick Questions:**")
        q_cols = st.columns(5)
        selected_prompt = None

        if q_cols[0].button("👤 Applicant Name"):
            selected_prompt = "What is the applicant name?"
        if q_cols[1].button("📅 Date of Birth"):
            selected_prompt = "What is the date of birth?"
        if q_cols[2].button("🏠 Full Address"):
            selected_prompt = "What is the address?"
        if q_cols[3].button("🆔 ID Number"):
            selected_prompt = "What is the enrollment or ID number?"
        if q_cols[4].button("📑 Required Docs"):
            selected_prompt = "What documents are required to be submitted?"

        with st.form("qa_form"):
            default_q = selected_prompt or st.session_state.get("last_question", "")
            user_question = st.text_input("Enter your question:", value=default_q, placeholder="e.g. What is the applicant's category?")
            ask_submitted = st.form_submit_button("Ask Form Assistant", type="primary")

        if ask_submitted and user_question.strip():
            st.session_state["last_question"] = user_question
            with st.spinner("Finding answer in document..."):
                st.session_state["qa_result"] = pipe.ask_with_evidence(
                    user_question,
                    result.ocr_text,
                    ocr_confidence=result.ocr_confidence,
                    en_context=result.translations.get("en"),
                    analysis=result.analysis,
                    fields=result.fields,
                    target_lang=target_lang,
                )

        if st.session_state.get("qa_result"):
            qa_res = st.session_state["qa_result"]
            st.markdown("### Answer:")
            st.success(qa_res["answer"])

            if qa_res.get("evidence"):
                st.markdown("##### 🔍 Evidence from Form:")
                for ev in qa_res["evidence"]:
                    st.markdown(f'<div class="instruction-item">"{ev}"</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 4: TRANSLATION
    # ==========================================
    with tab_translation:
        st.subheader(f"Document Translation (Target: {target_lang.upper()})")
        st.caption("Powered by locally fine-tuned NLLB-200 with Indian Government Administrative Glossary.")

        t_col1, t_col2 = st.columns(2)
        calc_height = min(750, max(380, len(result.ocr_text.splitlines()) * 26))
        with t_col1:
            st.markdown("#### 📄 Original Extracted Text")
            st.text_area("Source Text", result.ocr_text, height=calc_height, label_visibility="collapsed")
        with t_col2:
            st.markdown(f"#### 🌐 Translated Text ({target_lang.upper()})")
            trans_text = result.translations.get(target_lang, "")
            st.text_area("Translated", trans_text, height=calc_height, label_visibility="collapsed")

        if trans_text:
            st.download_button(
                label=f"📥 Download Translated Text ({target_lang.upper()})",
                data=trans_text,
                file_name=f"translated_form_{target_lang}.txt",
                mime="text/plain",
            )

    # ==========================================
    # TAB 5: NLP ANALYSIS (ACADEMIC / SYLLABUS)
    # ==========================================
    with tab_nlp:
        st.subheader("NLP Feature Analysis & Metrics")
        st.caption("Document linguistic analysis using Tokenization, Stopwords Removal, TF-IDF, and Lexical Diversity.")

        nlp = result.nlp_features
        metric_cols = st.columns(4)
        metric_cols[0].metric("Sentences", len(nlp.sentences))
        metric_cols[1].metric("Word Tokens", getattr(nlp, "token_count", len(nlp.tokens)))
        metric_cols[2].metric("Unique Vocabulary", getattr(nlp, "vocabulary_size", len(set(nlp.tokens))))
        lex_div = getattr(nlp, "lexical_diversity", 0.0)
        metric_cols[3].metric("Lexical Diversity", f"{lex_div:.1%}")

        st.markdown("---")
        nlp_c1, nlp_c2 = st.columns(2)
        with nlp_c1:
            st.markdown("##### 📊 Top TF-IDF Weighted Terms")
            st.dataframe(
                [{"Term": term, "TF-IDF Weight": round(w, 4)} for term, w in nlp.tfidf_terms],
                hide_index=True,
                use_container_width=True,
            )
        with nlp_c2:
            st.markdown("##### 🔤 Most Frequent Terms (Bag of Words)")
            st.dataframe(
                [{"Term": term, "Frequency": cnt} for term, cnt in nlp.bag_of_words],
                hide_index=True,
                use_container_width=True,
            )

    # ==========================================
    # DEVELOPER / ANNOTATION SECTION
    # ==========================================
    with st.expander("🛠️ Developer Tools: Line Labeling & Annotations (Offline Model Training)"):
        st.caption("Review extracted lines and assign training labels for local line-role classifier fine-tuning.")
        official_forms = sorted((ROOT / "data" / "raw" / "gov_forms").glob("*.pdf"))
        if not official_forms:
            st.info("Place sample form PDFs in data/raw/gov_forms to label.")
        else:
            selected_form = st.selectbox(
                "Source Form for Labeling",
                official_forms,
                format_func=lambda path: path.name,
                key="annotation_source_form",
            )
            authorized = st.checkbox("Authorized for local model training", key="annotation_authorized")
            if authorized:
                source_bytes = selected_form.read_bytes()
                source_id = document_key(selected_form.name, source_bytes)
                lines_key = f"annotation_lines_{source_id}"
                labels_key = f"annotation_labels_{source_id}"
                if lines_key not in st.session_state:
                    with st.spinner("Extracting lines..."):
                        st.session_state[lines_key] = extract_annotation_lines(selected_form, use_gpu=use_gpu)
                    st.session_state[labels_key] = ["unlabeled"] * len(st.session_state[lines_key])
                annotation_lines = st.session_state[lines_key]
                if annotation_lines:
                    with st.form("indian_form_line_annotations"):
                        annotation_table = pd.DataFrame({
                            "line": annotation_lines,
                            "label": st.session_state[labels_key],
                        })
                        edited_annotations = st.data_editor(
                            annotation_table,
                            hide_index=True,
                            use_container_width=True,
                            disabled=["line"],
                            key=f"annotation_editor_{source_id}",
                            column_config={
                                "line": st.column_config.TextColumn("Extracted line", width="large"),
                                "label": st.column_config.SelectboxColumn("Label", options=INDIAN_LINE_LABELS, required=True),
                            },
                        )
                        save_annotations = st.form_submit_button("Save Annotations")
                    if save_annotations:
                        selected_labels = edited_annotations["label"].fillna("unlabeled").tolist()
                        try:
                            saved = save_document_annotations(
                                ROOT / "data" / "annotations" / "indian_form_lines.jsonl",
                                source_id,
                                selected_form.name,
                                annotation_lines,
                                selected_labels,
                            )
                            st.session_state[labels_key] = selected_labels
                            st.success(f"Saved {saved} labeled lines locally.")
                        except ValueError as err:
                            st.error(str(err))
else:
    st.info("👆 Please upload a government form (PDF or image) to begin analysis.")
