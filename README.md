---
title: gift-finder
emoji: 🎁
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# gift-finder

> Track A — AI Engineering Intern take-home assessment.

## One-Paragraph Summary

I built `gift-finder`, a hosted AI gift finder for shoppers who need a quick, trustworthy gift shortlist for babies, toddlers, or new moms. A user can type natural language like "thoughtful gift for a friend with a 6-month-old, under 200 AED" or an Arabic equivalent, and the system retrieves catalog-grounded products, explains why they fit, validates the response against a schema, and refuses or asks for more detail when the catalog does not support a confident answer. The prototype uses Groq LLaMA 3.3 70B, local sentence-transformer embeddings, ChromaDB RAG, FastAPI, HTML/CSS, and Pydantic validation on a synthetic bilingual product catalog.

## Prototype Access

- Live demo: https://vaigandlahemanth-gift-finder.hf.space
- GitHub repo: https://github.com/VaigandlaHemanth/gift-finder
- 3-minute Loom: add the Loom recording link here before sending the final email
- Submission track: Track A — AI Engineering Intern

## Setup And Run

Prerequisites:
- Python 3.10+
- Free Groq API key from https://console.groq.com

Run locally:

```bash
git clone https://github.com/VaigandlaHemanth/gift-finder.git
cd gift-finder
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GROQ_API_KEY
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Run evals:

```bash
python eval_runner.py
```

First run may download `sentence-transformers/all-MiniLM-L6-v2` once. After that, the app rebuilds the small local Chroma index automatically.

## Architecture

```text
User query
  -> language + currency detection
  -> LLM constraint extraction
  -> ChromaDB retrieval over synthetic product catalog
  -> deterministic budget/age/recipient filtering
  -> LLM product ranking
  -> deterministic catalog grounding
  -> Pydantic schema validation
  -> same-language HTML response + transparency drawer
```

Key behavior:
- **Language selection**: English queries show English-only visible results. Arabic queries show Arabic-only visible results. Bilingual fields remain in the raw validated JSON for debugging and grading.
- **Currency selection**: AED/درهم queries stay in AED. INR/₹/rupees queries are converted to AED for filtering, then displayed back in approximate INR using a fixed prototype conversion rate.
- **Grounding**: product names, prices, categories, ages, stock, ratings, evidence bullets, and reasons are rebuilt from `data/products.json` after the LLM responds.
- **Hallucination control**: the LLM can rank retrieved products, but final user-facing product facts come only from catalog fields.
- **Uncertainty**: vague, unsupported, low-budget, age-out-of-scope, empty, pet, and adult/electronics queries return an explicit uncertainty/refusal note plus refinement suggestions.
- **Transparency**: the "See how the AI thought" drawer shows extracted constraints, behavior, confidence, grounding evidence, and raw validated JSON.

## Evaluation

Latest expanded result:

```text
20/20 cases passed
120/120 points
100.0%
```

Rubric:

| Criterion | Points | What It Checks |
|---|---:|---|
| Correct behavior | 1 | Recommends when appropriate and refuses when out of scope |
| Language match | 1 | Detected language matches expected language |
| Schema validity | 1 | Response validates against Pydantic models |
| Reasoning quality | 1 | Reasons are specific; uncertain cases say so |
| Grounding validity | 1 | Product IDs and evidence match `products.json` |
| Category accuracy | 1 | Recommendations match expected product categories |

The 20 cases include easy English/Arabic, vague queries, low-budget refusal, age-out-of-scope refusal, new-mom requests, mixed Arabic-English input, hallucination guardrails, and adversarial pet/adult-electronics inputs. Full details are in [EVALS.md](EVALS.md).

## Tradeoffs

I chose Gift Finder because it maps to a real e-commerce gift-discovery use case and hits the Track A AI engineering criteria without becoming a single-prompt demo: RAG, structured output validation, multilingual UX, tool-like constraint extraction, grounding, and evals. I intentionally did not build pediatric health triage because health advice is high-risk and harder to validate safely in a short take-home. I also skipped image input, user accounts, real inventory integrations, and A/B testing because they would add scope without improving the core grading signal.

Main technical choices:
- **Groq LLaMA 3.3 70B** for free, fast multilingual constraint extraction and re-ranking.
- **ChromaDB + all-MiniLM-L6-v2** for local retrieval with no extra reviewer API key.
- **FastAPI + HTML** instead of Streamlit so the custom UI can be hosted while keeping `GROQ_API_KEY` server-side.
- **Pydantic v2** so malformed JSON, missing uncertainty, or unsupported low-confidence responses fail explicitly.

Full architecture, failure modes, and production path are in [TRADEOFFS.md](TRADEOFFS.md).

## AI Usage Note

- Groq LLaMA 3.3 70B powers constraint extraction and product re-ranking.
- `all-MiniLM-L6-v2` creates local embeddings for ChromaDB retrieval.
- Claude/Codex were used as pair-coding assistants for scaffolding, refactors, UI polish, eval design, and documentation.
- I overruled generated code where grounding, same-language UX, or uncertainty needed deterministic guardrails.
- The important prompt rules are committed in `src/gift_finder.py`; eval scoring is in `eval_runner.py`.

## Time Log

- 45 min: problem selection, Track A scope, synthetic catalog review.
- 120 min: RAG pipeline, schema validation, grounding, uncertainty guardrails.
- 60 min: FastAPI backend, custom HTML UI, and Hugging Face hosting.
- 90 min: expanded 20-case eval suite, same-language UX, evidence checks.
- 45 min: README/EVALS/TRADEOFFS polish and Loom prep. Total: about 6 hours after hosting/eval polish.

## File Map

```text
app.py                 FastAPI backend and /api/find endpoint
Landing_page.html      Search page
Results_page.html      Same-language results UI and transparency drawer
data/products.json     165 synthetic bilingual catalog products
src/gift_finder.py     Constraint extraction, retrieval flow, grounding, uncertainty
src/schema.py          Pydantic response schemas
src/catalog.py         ChromaDB loading and semantic search
eval_runner.py         20-case eval runner with grounding checks
evals/eval_cases.json  Easy, hard, edge, and adversarial test cases
EVALS.md               Evaluation details and scores
TRADEOFFS.md           Architecture and product decisions
Dockerfile             Hugging Face Spaces deployment
```

## Final Submission

Email subject:

```text
AI Intern | Track A | Vaigandla Hemanth
```

Email body should contain one link:

```text
https://github.com/VaigandlaHemanth/gift-finder
```

Add the Loom link in the Prototype Access section before sending.
