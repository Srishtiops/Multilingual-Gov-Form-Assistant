# Form evaluation set

Store reviewed examples as UTF-8 JSON Lines in `indian_forms.jsonl` (one JSON object per line). Use public blank forms or data you are authorized to evaluate. Keep documents from the same source form in the same split; do not tune on the held-out set.

Each record can represent either reviewed extracted text (to score extraction/Q&A alone) or a local image/PDF path (to run the document through OCR and then score downstream tasks). For OCR evaluation, include `ocr_reference` copied and reviewed against the source image. Use scanned pages or image files when measuring OCR; digitally embedded PDF text bypasses OCR. Add only reviewed expected values and answers.

```json
{"id":"sample-form-page-1","language":"en","context":"Applicant Name: Example Person\nDate of Birth: 01/02/1990","ocr_reference":"Applicant Name: Example Person\nDate of Birth: 01/02/1990","fields":{"name":"Example Person","date_of_birth":"01/02/1990"},"questions":[{"question":"What is the applicant name?","answer":"Example Person"},{"question":"What is the application fee?","answer":null}]}
```

For an image/PDF example, replace `context` with an `image` path such as `data/evaluation/images/form_page1.png`; paths are relative to the project root. `answer: null` means the answer is absent and the assistant should abstain. Do not use synthetic/sample entries as evidence of real-world accuracy. A useful first held-out set has multiple independent forms per language and includes answerable and unanswerable questions. Record reviewers and source URLs in a separate manifest, and redact personal information.

Run `python scripts/evaluate_forms.py --dataset data/evaluation/indian_forms.jsonl`. The script reports OCR character similarity when references are present, field exact-match precision/recall/F1, and QA exact match/F1 plus abstention accuracy. Q&A invokes the configured local model and can take time on CPU. It writes per-example errors for review.
