# Training data for English, Hindi, and Marathi

This project targets **en**, **hi**, and **mr**. Use the scripts below to fetch public corpora; add your own labeled Indian form lines in the app (**Training labels**).

## Parallel text (translation fine-tuning)

| Source | Languages | HF / URL | Use |
|--------|-----------|----------|-----|
| **Samanantar** | hi↔en, mr↔en | `ai4bharat/samanantar` | Main Indic parallel corpus; download with `scripts/download_indic_training_data.py` |
| **Flores-200** | en, hi, mr dev/test | `facebook/flores` | Held-out BLEU for translation; same script |
| **OPUS-100** | en–hi | `Helsinki-NLP/opus-100` | Extra en↔hi pairs (sample) |
| **Government glossary** | en, hi, mr | `data/glossary/government_forms.json` | Domain terms; exported as parallel rows by `build_glossary.py` |
| **IndicTrans training packs** | many indic | [AI4Bharat IndicTrans2 data](https://github.com/AI4Bharat/indicTrans2) | Full-scale training (large; request separately) |

Marathi parallel data is smaller than Hindi. Combine Samanantar **mr** with glossary terms and, when possible, manually aligned snippets from official UIDAI / Passport Seva handbooks under `data/raw/gov_forms/`.

## Document understanding (line roles)

| Source | Languages | Location | Use |
|--------|-----------|----------|-----|
| **FUNSD** | en | `data/raw/funsd` | Generic form layout; `train_form_role_models.py --task form-role` |
| **XFUND** | de, es, fr, … | `data/raw/xfund` | Multilingual layout (not Indian scripts); same script |
| **Your labels** | en, hi, mr | `data/annotations/indian_form_lines.jsonl` | **Target domain**; `train_form_role_models.py --task indian` |

## Question answering

| Source | Languages | HF | Use |
|--------|-----------|-----|-----|
| **TyDi QA** | hi (+ others) | `google-research-datasets/tydiqa` | Reading comprehension; exported to `data/training/qa/` |
| **Form eval set** | en, hi, mr | `data/evaluation/indian_forms.jsonl` | Human-reviewed metrics only |

## OCR

EasyOCR ships pretrained **en** and **hi** (Devanagari). Marathi forms use the same script; the app maps **mr → hi** for OCR. For higher Marathi accuracy, consider [IndicOCR / docTR](https://github.com/AI4Bharat/indic-doc) as a future engine.

## LLM (optional, no fine-tune required)

- **Ollama + Qwen 2.5 3B** (see README): stronger grounded Q&A on long form text.
- Fine-tuning an LLM on form Q&A is optional; start with TyDi QA + your `indian_forms.jsonl` Q&A pairs if you add a `train_qa` script later.

## Recommended workflow

```powershell
cd D:\NLP\gov_form_assistant
.\.venv-gpu\Scripts\pip.exe install -e .
python scripts\download_datasets.py
python scripts\download_indic_training_data.py --max-rows 50000
python scripts\build_glossary.py
python scripts\train_translation.py --device auto --max-steps 800
python scripts\train_form_role_models.py --task form-role --device auto
# After labeling ≥5 official forms in the app:
python scripts\train_form_role_models.py --task indian --device auto
python scripts\evaluate_forms.py --device auto
```

Keep evaluation JSONL **disjoint** from training annotations and parallel corpora used for fine-tuning.
