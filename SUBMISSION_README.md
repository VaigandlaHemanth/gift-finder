# Mumzworld AI Intern | Track A | Vaigandla Hemanth Kumar

---

## Track

**Track A — AI Engineering**

---

## What I Built

`gift-finder` is an AI-powered gift recommendation tool for Mumzworld shoppers. You type something like *"thoughtful gift for a friend with a 6-month-old, under 200 AED"* — or the same in Arabic — and the system retrieves catalog-grounded products, explains why each fits, and refuses or asks for clarification when it can't give a confident answer. It handles bilingual input (EN/AR), detects currency (AED vs INR), grounds every product fact against a 165-item synthetic catalog so the LLM can't hallucinate details, and validates the full response through a Pydantic schema before anything reaches the user.

---

## Prototype Access

- **Live demo:** https://vaigandlahemanth-gift-finder.hf.space  
- **GitHub repo:** https://github.com/VaigandlaHemanth/gift-finder  
- **3-minute walkthrough drive:** https://drive.google.com/file/d/1Wv8zs3jv10B806B__JHLlRXLEx0W2Iqp/view?usp=sharing

### Setup (local)

```bash
git clone https://github.com/VaigandlaHemanth/gift-finder.git
cd gift-finder
pip install -r requirements.txt
cp .env.example .env
# set GROQ_API_KEY in .env
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

Run evals:

```bash
python eval_runner.py
```

First run downloads `all-MiniLM-L6-v2` once, then rebuilds the Chroma index automatically.

---

## EVALS.md

### Latest Score

```
20/20 cases passed
120/120 points (100%)
```

### Rubric (6 points per case)

| Criterion | What it checks |
|---|---|
| Correct behavior | Recommends when it should, refuses when it shouldn't |
| Language match | Output language matches input language |
| Schema validity | Response validates against Pydantic models |
| Reasoning quality | Reasons are specific; uncertain cases say so |
| Grounding validity | Product IDs and evidence match `data/products.json` |
| Category accuracy | Recommendations fall in expected product categories |

### Test Suite (20 cases)

**Easy**
- EVAL-001: gift for newborn under 100 AED → Recommend (Clothing, Bath, Books, Feeding)
- EVAL-002: هدية لطفل عمره 6 أشهر أقل من 150 درهم → Recommend in Arabic

**Edge**
- EVAL-003: "something nice for a baby" → Recommend with low-confidence flag
- EVAL-004: gift under 10 AED → Refuse (cheapest item is 35 AED)
- EVAL-005: gift for a new mom recovering from birth → Recommend Postpartum
- EVAL-006: gift for a 5-year-old → Refuse (scope is 0–36 months)
- EVAL-016: "a nice gift" → Recommend cautiously with refinement suggestions
- EVAL-018: هدية حلوة للبيبي → Arabic vague query; Arabic-only suggestions

**Hard**
- EVAL-007: educational toy, 18-month-old, under 200 AED → multi-constraint
- EVAL-008: Arabic implicit age (1.5 years = 18 months), practical preference
- EVAL-010: Arabic luxury, twins, postpartum timing
- EVAL-011: travel stroller, lightweight, compact
- EVAL-013: teething and cranky baby, something soothing
- EVAL-015: nursing pillow + breast pump, 2 weeks postpartum
- EVAL-017: "organic bamboo teether under 50 AED" — hallucination guard; must not invent material facts not in catalog
- EVAL-019: mixed Arabic-English input → still produces Arabic UX

**Adversarial**
- EVAL-009: gift for my dog → Refuse
- EVAL-012: empty query → Refuse
- EVAL-014: أقل من 30 درهم → Arabic low-budget refuse
- EVAL-020: smartphone for my adult brother → Refuse (out of scope)

### How Evals Work

`eval_runner.py` loads the product catalog into ChromaDB, runs all 20 cases through the full pipeline, scores each against the rubric, and writes `evals/eval_report.json`.

**Passing thresholds:**
- ≥80%: Strong prototype, ready for submission
- 60–80%: Needs prompt tuning or edge case work
- <60%: Critical issues in constraint extraction or retrieval

### Common failure patterns to watch

1. Arabic transliteration — Latin characters slip into Arabic fields; fix with prompt + validator
2. Generic reasons — "Good product" instead of specific features; tighten prompt
3. Wrong refusal — recommending for out-of-scope queries; improve constraint extraction
4. Schema violations — missing fields or wrong types; check Pydantic validators

---

## TRADEOFFS.md

### Why this problem

Gift Finder maps directly to Mumzworld's core use case and hits all Track A AI engineering criteria in one place: RAG, structured output validation, multilingual UX, tool-like constraint extraction, hallucination grounding, and a real eval suite. I skipped pediatric health triage because health advice is high-risk and harder to validate safely in a short take-home. Image input, user accounts, real inventory integration, and A/B testing were also out of scope — they add surface area without improving the core grading signal.

### Stack choices

**Groq LLaMA 3.3 70B** — free, fast, multilingual. Good enough for constraint extraction and re-ranking without burning budget.

**ChromaDB + all-MiniLM-L6-v2** — local retrieval, no extra API key for reviewers, rebuilds automatically on first run.

**FastAPI + plain HTML** instead of Streamlit — lets me host on Hugging Face Spaces with `GROQ_API_KEY` server-side rather than exposing it in the browser.

**Pydantic v2** — malformed JSON, missing uncertainty, or unsupported confidence fails loudly. Silent failures are worse than loud ones.

### Grounding strategy

The LLM ranks retrieved products but never writes the final product card. After the LLM responds, deterministic code rebuilds product names, prices, categories, age ranges, stock status, ratings, evidence bullets, and reasons from `data/products.json`. The user never sees a fact that isn't in the catalog.

### Language and currency

Language detection is based on character sets and keywords. English input produces English-only visible results. Arabic input produces Arabic-only visible results. Bilingual fields stay in the raw JSON for debugging. Currency detection: AED/درهم stays AED; INR/₹/rupees converts to AED for filtering, then converts back to approximate INR for display using a fixed prototype rate.

### Uncertainty handling

Vague, unsupported, low-budget, age-out-of-scope, empty, pet, and adult/electronics queries return an explicit uncertainty note and refinement suggestions rather than a low-confidence product list.

### What I'd do with more time

- Real Mumzworld product catalog via API instead of synthetic data
- User session history for follow-up queries ("show me something cheaper")
- Image upload — user photos a nursery, system suggests matching décor gifts
- A/B test prompt variants against the eval rubric automatically
- Arabic character-level validation in the Pydantic schema

### Failure modes I know about

- Very long mixed-language queries can confuse language detection
- The fixed INR conversion rate (≈0.044) drifts with actual exchange rates
- ChromaDB is in-memory; scale would need a persistent vector store
- The synthetic catalog has 165 items; recall degrades on niche queries

---

## AI Usage Note

- **Groq LLaMA 3.3 70B** — constraint extraction and product re-ranking at inference time
- **all-MiniLM-L6-v2** — local embeddings for ChromaDB semantic retrieval
- **Claude (Anthropic)** — pair-coding for scaffolding, refactors, UI polish, eval design, and documentation drafts
- I overruled generated code wherever grounding, same-language UX, or uncertainty needed deterministic logic
- Core prompt rules live in `src/gift_finder.py`; eval scoring logic is in `eval_runner.py`

---

## Time Log

- ~45 min: problem selection, Track A scope, synthetic catalog design
- ~120 min: RAG pipeline, schema validation, grounding, uncertainty guardrails
- ~60 min: FastAPI backend, HTML UI, Hugging Face hosting
- ~90 min: 20-case eval suite, same-language UX, evidence checks
- ~45 min: README/EVALS/TRADEOFFS polish and Loom prep
- **Total: ~6 hours** (went slightly over the 5-hour guideline due to hosting and eval expansion)

---

## File Map

```
app.py                  FastAPI backend and /api/find endpoint
Landing_page.html       Search page
Results_page.html       Same-language results UI and transparency drawer
data/products.json      165 synthetic bilingual catalog products
src/gift_finder.py      Constraint extraction, retrieval, grounding, uncertainty
src/schema.py           Pydantic response schemas
src/catalog.py          ChromaDB loading and semantic search
eval_runner.py          20-case eval runner with grounding checks
evals/eval_cases.json   Easy, hard, edge, and adversarial test cases
EVALS.md                Evaluation details and scores
TRADEOFFS.md            Architecture and product decisions
Dockerfile              Hugging Face Spaces deployment
```

---

*Submission by Vaigandla Hemanth — B.Tech CSE (AI & ML), Amrita Vishwa Vidyapeetham, 2026*
