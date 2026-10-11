# Multilingual Indian Government Form Assistant

An end-to-end NLP assistant designed to process, translate, and explain Indian government forms and applications (Aadhaar, PAN, Driving Licence, Passport, Voter ID, Ration Card, RTI, and educational forms) across **English**, **Hindi (हिंदी)**, and **Marathi (मराठी)**.

Built to run **100% offline** on local consumer hardware (e.g., RTX 4050 Laptop GPU / 16 GB RAM).

---

## ✨ Features

- **Multilingual OCR Engine**: Extracts text from scanned PDFs and images in English and Devanagari scripts (Hindi and Marathi) with line confidence scoring.
- **Fine-Tuned Domain Translation**: Uses a local fine-tuned sequence-to-sequence model (`facebook/nllb-200-distilled-600M`) augmented with an official Indian administrative glossary. Preserves original line breaks, blank spaces, and layout structure across `en` ↔ `hi`, `en` ↔ `mr`, and `hi` ↔ `mr`.
- **Structured Field Extraction**: Accurately detects and normalizes:
  - **Personal Details**: Applicant Full Name, Parent/Guardian/Spouse Name, Date of Birth, Gender, Category/Caste.
  - **Identification & Contact**: Document ID/Enrollment No., Mobile Number, Email ID, Residential/Permanent Address.
- **Checklist & Instruction Guidance**: Automatically extracts mandatory rules, form-filling directions, and required supporting documents (Proof of Identity, Address Proof, Certificates, Annexures), rendered in the citizen's selected language.
- **Grounded Q&A Assistant**: An interactive Q&A system powered by Google Flan-T5 and domain rules. Answers are grounded in the form text with verified source quotes to eliminate hallucinations; includes blank template detection.
- **Modern Responsive UI**: Built with Streamlit, supporting both Dark and Light themes with high-contrast cards, quick-question pills, and downloadable translations.

---

## 🛠️ Technology Stack

| Layer | Component / Model | Description |
| :--- | :--- | :--- |
| **OCR** | EasyOCR (`en`, `hi`) | Text detection and recognition for Latin and Devanagari scripts |
| **Translation** | Fine-Tuned NLLB-200 (600M) | Domain-adapted seq2seq translation with trilingual administrative glossary |
| **Document Understanding** | Regex + Role Classifier | Form classification, instruction detection, and requirement extraction |
| **Question Answering** | Google Flan-T5-Base | Zero-shot grounded reasoning over extracted form text and translated context |
| **Web Interface** | Streamlit | Responsive dashboard with field cards, checklists, and side-by-side translation |
| **Frameworks** | PyTorch, Hugging Face Transformers | GPU-accelerated local inference (CUDA) |

---

## 📂 Repository Structure

```
gov_form_assistant/
├── app/
│   └── streamlit_app.py              # Main Streamlit web application
├── src/gov_form_assistant/
│   ├── document_io.py                # PDF and image ingestion (PyMuPDF / PIL)
│   ├── ocr.py                        # EasyOCR integration with confidence scoring
│   ├── translation.py                # NLLB-200 translator + layout-preserving document translation
│   ├── field_extraction.py           # Multilingual field extraction rules and regex
│   ├── form_understanding.py         # Document classifier and instruction extractor
│   ├── qa.py                         # Grounded Q&A engine with Flan-T5 & Ollama fallback
│   ├── preprocessing.py              # Multilingual tokenization, stopwords, TF-IDF
│   └── pipeline.py                   # Unified end-to-end assistant pipeline
├── scripts/
│   ├── build_marathi_training_corpus.py  # Builds aligned Marathi-English & Hindi-Marathi datasets
│   ├── build_glossary.py                 # Compiles trilingual government glossary (684+ terms)
│   ├── train_translation.py              # Fine-tunes NLLB-200 seq2seq model on GPU
│   └── train_form_role_models.py         # Trains document line classifier
├── data/
│   ├── glossary/government_forms.json   # Trilingual administrative glossary
│   ├── training/parallel/               # Parallel JSONL datasets for fine-tuning
│   └── annotations/                     # Annotated Indian government form lines
├── models/
│   ├── huggingface/                     # Cached local model weights
│   └── training/nllb_gov_domain/        # Fine-tuned translation checkpoint
├── tests/                               # Automated pytest suite
├── config.yaml                          # Central project and model configuration
└── requirements.txt                     # Python package dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Python 3.11 or 3.12 recommended)
- NVIDIA GPU with CUDA support (optional, but recommended for faster inference)

### 2. Installation

Clone the repository and set up a virtual environment:

```powershell
# Clone the repository
git clone https://github.com/your-username/gov-form-assistant.git
cd gov-form-assistant

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configuration & Offline Mode

The application is pre-configured in `config.yaml` to run using local model weights (`local_files_only: true`), ensuring zero network dependency during inference.

If you are running the project for the first time without cached Hugging Face models, download the base weights:

```powershell
python scripts/download_models.py
```

### 4. Running the Web Application

Launch the Streamlit dashboard:

```powershell
streamlit run app/streamlit_app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🔄 Model Training & Fine-Tuning

### Fine-Tuning the Translation Model
To retrain or update the domain-adapted NLLB model on custom parallel data:

```powershell
# 1. Build balanced Marathi & Hindi parallel corpus
python scripts/build_marathi_training_corpus.py

# 2. Run GPU fine-tuning (saves checkpoint to models/training/nllb_gov_domain)
python scripts/train_translation.py --device cuda --batch-size 4 --max-steps 80 --max-rows 3500
```

### Training the Indian Form Line Classifier
```powershell
python scripts/train_form_role_models.py --task indian --device auto
```

---

## 🧪 Testing

Run automated unit and integration tests:

```powershell
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. Base model checkpoints (`facebook/nllb-200-distilled-600M`, `google/flan-t5-base`) are subject to their respective open-source licenses by Meta and Google.
