# Auto Email / Ticket Categorizer

A lightweight NLP classifier that routes an incoming support ticket to the
correct department in real time: **BILLING**, **TECHNICAL**, **HR**, or **GENERAL**.
Low-confidence or ambiguous tickets are automatically escalated to **NEEDS HUMAN REVIEW**.

## Files

- `tickets_dataset.csv` — dummy labeled dataset (240 tickets, 60 per category, no duplicates)
- `ticket_classifier.py` — cleans data, trains/compares models, evaluates, and demos
- `ticket_router_model.joblib` — saved trained model (generated after running the script)
- `api.py` — FastAPI wrapper: serves the trained model over HTTP and hosts the web UI
- `static/` — the **Routing Desk** web UI (`index.html`, `styles.css`, `app.js`)
- `requirements.txt` — all dependencies for both the script and the web app

## How to run

> **Note:** This project uses a Python virtual environment (`.venv/`) to keep
> dependencies isolated from the system Python. Follow the section for your OS below.

---

### 🐧 Linux / 🍎 macOS

**First-time setup:**

```bash
cd ~/ticket_categorizer
python3 -m venv .venv
.venv/bin/pip install pandas scikit-learn joblib
```

**Run — train, evaluate, and 5-ticket demo:**

```bash
.venv/bin/python3 ticket_classifier.py
```

**Run with interactive CLI:**

```bash
.venv/bin/python3 ticket_classifier.py --demo
```

**Or activate the venv first (so you can just type `python3`):**

```bash
source .venv/bin/activate
python3 ticket_classifier.py            # train + evaluate + demo
python3 ticket_classifier.py --demo    # + interactive CLI
deactivate                              # exit the venv when done
```

---

### 🪟 Windows (Command Prompt)

**First-time setup:**

```cmd
cd %USERPROFILE%\ticket_categorizer
python -m venv .venv
.venv\Scripts\pip install pandas scikit-learn joblib
```

**Run — train, evaluate, and 5-ticket demo:**

```cmd
.venv\Scripts\python ticket_classifier.py
```

**Run with interactive CLI:**

```cmd
.venv\Scripts\python ticket_classifier.py --demo
```

**Or activate the venv first:**

```cmd
.venv\Scripts\activate
python ticket_classifier.py            :: train + evaluate + demo
python ticket_classifier.py --demo    :: + interactive CLI
deactivate                             :: exit the venv when done
```

---

### 🪟 Windows (PowerShell)

**First-time setup:**

```powershell
cd $env:USERPROFILE\ticket_categorizer
python -m venv .venv
.venv\Scripts\pip install pandas scikit-learn joblib
```

**Run — train, evaluate, and 5-ticket demo:**

```powershell
.venv\Scripts\python ticket_classifier.py
```

**Run with interactive CLI:**

```powershell
.venv\Scripts\python ticket_classifier.py --demo
```

**Or activate the venv first:**

```powershell
.venv\Scripts\Activate.ps1
python ticket_classifier.py            # train + evaluate + demo
python ticket_classifier.py --demo    # + interactive CLI
deactivate                             # exit the venv when done
```

> **PowerShell tip:** If you get an "execution policy" error when activating,
> run this once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Web UI — the Routing Desk

The same model, wrapped in FastAPI and served with a small front end so a ticket
can be routed in the browser instead of the terminal.

The server loads `ticket_router_model.joblib` at startup and trains a fresh
model only if that file is missing.

### 🐧 Linux / 🍎 macOS

```bash
cd ~/ticket_categorizer
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --reload
```

### 🪟 Windows (Command Prompt)

```cmd
cd %USERPROFILE%\ticket_categorizer
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn api:app --reload
```

### 🪟 Windows (PowerShell)

```powershell
cd $env:USERPROFILE\ticket_categorizer
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn api:app --reload
```

> **No `.venv` yet?** Create it first with the **First-time setup** block for your
> OS above, then run `pip install -r requirements.txt` instead of the package list
> — `requirements.txt` covers both the script and the web app.

**Then open <http://127.0.0.1:8000> in a browser.**

The last command does not finish: it prints `Uvicorn running on
http://127.0.0.1:8000` and keeps running, because it *is* the server. Leave that
terminal open while you use the page, and press `Ctrl+C` in it to stop.
`--reload` restarts the server automatically whenever you edit the code.

The page prints a **routing slip** for each ticket: the department, the
confidence, and — the point of the screen — the two checks that actually decide
whether the ticket is routed or held. The bar chart shows the model's
probability across all four departments with the 50% confidence floor drawn
across it, so a held ticket visibly falls short of the line. Colour marks the
*decision* (cleared / held), never the department; every bar is labelled in
words.

### Endpoints

| Method | Path | Does |
|---|---|---|
| `GET` | `/` | The Routing Desk UI |
| `GET` | `/api/health` | Model name, departments, and the two thresholds |
| `GET` | `/api/samples` | The 5 unseen sample tickets |
| `POST` | `/api/classify` | Route one ticket |
| `POST` | `/api/classify/batch` | Route up to 100 tickets in one call |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

```bash
curl -X POST http://127.0.0.1:8000/api/classify \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text": "My card was charged twice for the same order, please refund"}'
```

```json
{
  "id": "TCK-3001",
  "department": "BILLING",
  "model_prediction": "BILLING",
  "confidence": 0.9422,
  "needs_review": false,
  "review_reason": null,
  "runner_up": {"department": "HR", "confidence": 0.0208},
  "priority": "NORMAL",
  "distribution": {"BILLING": 0.9422, "GENERAL": 0.0183, "HR": 0.0208, "TECHNICAL": 0.0187}
}
```

Adding `?ticket=...` to the page URL routes that ticket on load, so a particular
slip can be linked to directly.

## Approach summary (for the submission form)

Used TF-IDF (unigrams + bigrams) with explicit text cleaning, then trained
and compared Multinomial Naive Bayes and Logistic Regression, keeping the
better performer. Added two human-review escalation rules:

1. **Low confidence** — tickets where the top prediction scores below 50% are
   routed to "needs human review" instead of being auto-assigned.
2. **Ambiguous category** — tickets where the top-2 predictions are within 20%
   of each other (e.g. BILLING 52% vs TECHNICAL 41%) are also escalated, since
   a near-tie means the model can't reliably distinguish the category.
   A keyword-based urgent/normal priority tag is layered on top of every prediction,
   independent of the ML model.

> Note: the thresholds were originally 60% / 20%, but that pairing made the
> ambiguity check mathematically unreachable on its own (with 4 classes
> summing to 1, top1 &ge; 60% forces the gap to already be &ge; 20%). Lowered
> the confidence threshold to 50% so the ambiguity check has real room to
> fire independently — see `assess_review()` and `test_review_logic()` in
> `ticket_classifier.py`.

## Approach in detail

1. **Preprocessing**: `clean_text()` lowercases, strips punctuation/symbols,
   and collapses whitespace before vectorizing — kept explicit and separate
   from the vectorizer so it's easy to extend (stemming, lemmatization, etc.)
2. **Features**: `TfidfVectorizer` (unigrams + bigrams, English stop words
   removed). Bigrams help capture short domain phrases like "500 error" or
   "leave balance".
3. **Model choice — tested, not assumed**: both **Multinomial Naive Bayes**
   (the classic fast baseline for text, works well on small/sparse TF-IDF
   data) and **Logistic Regression** (models feature weights more flexibly)
   are trained and evaluated; the script automatically keeps whichever
   scores higher on the held-out test split.
4. **Evaluation**: stratified train/test split, accuracy, per-class
   precision/recall/F1, and a confusion matrix.
5. **Real-time routing**: `route_ticket()` returns a department, a
   confidence score, the runner-up prediction, and:
   - **Bonus — confidence score output**: returned alongside every prediction.
   - **Bonus — low-confidence review threshold**: if top confidence < 50%,
     the ticket is routed to `NEEDS HUMAN REVIEW`.
   - **Bonus — ambiguity review threshold**: if the gap between the top-2
     predictions is < 20%, the ticket is also routed to `NEEDS HUMAN REVIEW`
     (e.g. a billing ticket that also describes a technical error). This is
     unit-tested independently of the trained model in `test_review_logic()`,
     since on this dataset the trained model's confidence and margin turned
     out to be highly correlated (r &asymp; 0.98) — see the reflection note.
   - **Bonus — review reason**: every escalated ticket includes a plain-text
     `review_reason` field explaining why it was flagged.
   - **Bonus — priority tagging**: a keyword rule (e.g. "urgent", "down",
     "not working", "crash") flags a ticket `URGENT` vs `NORMAL`,
     independent of the ML prediction.
6. **Bonus — 5 new unseen tickets**: `run_static_demo()` predicts on 5 tickets
   written fresh for this assessment (not in the training data), including
   one deliberate edge case (gibberish text) to show the review threshold
   catching a ticket the model can't confidently place.
7. **Bonus — mini live demo**: run with `--demo` flag for an interactive CLI
   where you can type a ticket and get an instant routing decision.
8. **Bonus — reflection note**: included at the bottom of `ticket_classifier.py`.

## Output format

Each routed ticket now returns:

| Field                | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `department`       | Final routed department (or`NEEDS HUMAN REVIEW`)   |
| `model_prediction` | The model's top predicted category                   |
| `confidence`       | Probability of the top prediction                    |
| `runner_up`        | Second-best category and its probability             |
| `needs_review`     | `True` if escalated for any reason                 |
| `review_reason`    | Human-readable explanation (`None` if auto-routed) |
| `priority`         | `URGENT` or `NORMAL` based on keyword matching   |

### Human review triggers

```
needs_review = True  if:
  confidence < 0.50                        (low certainty)
  OR
  top1_confidence - top2_confidence < 0.20 (ambiguous boundary)
```

## Reflection note — what I'd improve with more data or time

1. More data, especially near category boundaries (e.g. a billing ticket
   that also mentions a technical error) — the confusion matrix shows most
   misclassifications happen around GENERAL, which is expected since it's
   a catch-all/ambiguous bucket compared to the other three.
2. Try a small fine-tuned transformer (e.g. DistilBERT) once there's enough
   data to justify it — TF-IDF + linear models plateau once ticket phrasing
   gets more varied or multilingual.
3. Replace the fixed 50% confidence threshold and 20% ambiguity gap with
   per-class tuned thresholds, and calibrate probabilities properly
   (`CalibratedClassifierCV`) since raw `predict_proba` from these models
   isn't always well-calibrated.
4. Expand priority tagging beyond keyword matching — a second lightweight
   classifier trained on urgency labels would generalize better than a
   fixed keyword list.
5. For truly multi-category tickets, explore a multi-label classification
   setup where a single ticket can be assigned to more than one department
   simultaneously.
6. While tuning the review thresholds, found that this model's top-1
   confidence and its margin over the runner-up are extremely correlated
   on this dataset (r &asymp; 0.98) — it's essentially never "confident but
   conflicted". The ambiguity check is unit-tested independently
   (`test_review_logic()`) to prove the logic is correct even though this
   particular dataset rarely exercises it in practice; with more time I'd
   want real examples that separate the two failure modes, or a noisier
   dataset where they naturally diverge.
