# Mumzworld AI Gift Finder

> AI-native gift recommendation engine for the Middle East's largest motherhood e-commerce platform.
> Built for the Mumzworld AI Engineering Intern take-home assessment.

## One-Paragraph Summary

An intelligent, bilingual (English/Arabic) gift finder that transforms natural language queries like "thoughtful gift for a friend with a 6-month-old, under 200 AED" into curated, structured recommendations with native Arabic copy, confidence scores, and graceful uncertainty handling. Built with Groq LLaMA 3.3 70B, local sentence-transformer embeddings, ChromaDB RAG, FastAPI, HTML/CSS, and Pydantic schema validation — all 100% free tier.

## Setup (Under 5 Minutes)

### Prerequisites
- Python 3.10+
- A free Groq API key ([get one here](https://console.groq.com))

### Step 1: Clone and Install
```bash
git clone <your-repo-url>
cd mumzworld-gift-finder
pip install -r requirements.txt
```

> **Note:** First run downloads the `all-MiniLM-L6-v2` embedding model (~90MB). This is cached after first use.

### Step 2: Configure API Key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Step 3: Run
```bash
# Launch the HTML web app
uvicorn app:app --reload

# Or run evaluations
python eval_runner.py
```

The app runs at `http://127.0.0.1:8000`.

## Free Hosting

Streamlit is not required. This repo uses the custom `Landing_page.html` and `Results_page.html` files as the frontend, with FastAPI as a small backend so the Groq key stays server-side.

Recommended free deployment:

1. Create a Hugging Face Space.
2. Choose Docker as the Space SDK.
3. Add `GROQ_API_KEY` as a Space secret.
4. Push this repo. The included `Dockerfile` starts the app on port `7860`.

Alternative: Render free web service with start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Architecture

```
User Query (EN or AR)
    ↓
Language Detection (langdetect + Arabic Unicode heuristic)
    ↓
Constraint Extraction (LLM → structured JSON: budget, age, occasion, recipient)
    ↓
Embedding-based Retrieval (ChromaDB + sentence-transformers, top-15 candidates)
    ↓
Constraint Filtering (budget ±10%, age ±2 months, recipient matching)
    ↓
LLM Re-ranking + Reasoning (Groq LLaMA 3.3 70B generates bilingual reasons)
    ↓
Structured Output Validation (Pydantic v2 schema + uncertainty validators)
    ↓
HTML Results Page (cards + refusal state + raw JSON drawer)
```

### Reliability Guardrails

1. **Retrieval vs generation**: retrieval finds matching products from `data/products.json`; generation only ranks/explains those retrieved products.
2. **Grounding**: the final response is post-processed against retrieved catalog IDs. Product names, prices, categories, age ranges, stock status, and reasons are overwritten from product data.
3. **Hallucination control**: the model is instructed not to invent features, and final reasons are rebuilt from catalog fields instead of trusting free-form LLM claims.
4. **Structured output**: every response validates through the Pydantic `GiftFinderResponse` schema before reaching the UI.
5. **Uncertainty**: unsupported, vague, low-budget, age-out-of-scope, and non-baby/mom requests return an explicit "I do not know / need more info" style note in English and Arabic.

### Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | Groq LLaMA 3.3 70B | Free tier, fast inference, strong multilingual Arabic |
| Embeddings | `all-MiniLM-L6-v2` local | Zero API keys, fast, sufficient for 165 products |
| Vector DB | ChromaDB local | Zero config, runs locally, under-5-min setup |
| Schema | Pydantic v2 | Strict validation, custom validators for Arabic quality |
| UI | HTML + FastAPI | Uses the custom landing/results pages while keeping the API key server-side |

## Evaluation

### Rubric

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correct Behavior | 20% | Recommends when appropriate, refuses when out of scope |
| Language Match | 20% | Detects input language, responds in same language |
| Schema Validity | 20% | Output validates against Pydantic schema |
| Reasoning Quality | 20% | Reasons are specific, not generic; Arabic is native not translated |
| Category Accuracy | 20% | Matches expected categories for constrained queries |

### Test Cases (15 total)

| ID | Query | Type | Expected |
|----|-------|------|----------|
| EVAL-001 | gift for newborn, under 100 AED | Easy EN | Recommend |
| EVAL-002 | هدية لطفل عمره 6 أشهر، أقل من 150 درهم | Easy AR | Recommend |
| EVAL-003 | something nice for a baby | Edge | Recommend (low confidence) |
| EVAL-004 | gift under 10 AED | Edge | Refuse (budget too low) |
| EVAL-005 | gift for a new mom recovering from birth | Edge | Recommend (postpartum) |
| EVAL-006 | gift for a 5-year-old | Edge | Refuse (age out of scope) |
| EVAL-007 | educational toy for 18-month-old, under 200 AED | Hard EN | Recommend |
| EVAL-008 | هدية لصديقتي عندها طفل عمره سنة ونص | Hard AR | Recommend |
| EVAL-009 | gift for my dog | Adversarial | Refuse |
| EVAL-010 | أريد هدية فاخرة لأختي التي أنجبت توأم | Hard AR | Recommend |
| EVAL-011 | best stroller for traveling to Dubai | Hard EN | Recommend |
| EVAL-012 | (empty query) | Adversarial | Refuse |
| EVAL-013 | gift for teething baby, something soothing | Hard EN | Recommend |
| EVAL-014 | أقل من 30 درهم | Adversarial AR | Refuse |
| EVAL-015 | nursing pillow and breast pump bundle | Hard EN | Recommend |

### Running Evals
```bash
python eval_runner.py
```

Results are saved to `evals/eval_report.json` with detailed results.

## Tooling & AI Assistance

### Models & APIs Used
- **Groq LLaMA 3.3 70B**: Primary LLM for constraint extraction, reasoning, and structured generation. Free tier via Groq console.
- **sentence-transformers/all-MiniLM-L6-v2**: Local embeddings for product catalog retrieval. Zero API cost.

### How I Used AI Tools
- **AI coding assistants (Claude/Codex)**: Architecture design, code scaffolding, dataset generation, eval case design, debugging, and documentation. Used as pair-coding partners for rapid iteration.
- **LLM via Groq API**: The product itself uses LLaMA 3.3 70B for all NLP tasks. Prompts were iterated 3-4 times to get Arabic output that feels native rather than translated.

### What Worked
- Groq's free tier is genuinely fast and capable for this scope
- Pydantic v2 schema validation catches silent failures (empty strings, wrong types) immediately
- Local embeddings + ChromaDB means zero external dependencies for reviewers

### What Didn't
- Initial Arabic prompts produced transliterated text (Latin characters in Arabic fields). Fixed by adding explicit "generate natively in Arabic, do not translate" instructions and a Pydantic validator that rejects Latin characters in Arabic fields.
- First constraint extraction prompt was too verbose, causing JSON parsing errors. Fixed by simplifying to 9 explicit fields and lowering temperature to 0.1.

### Prompts That Mattered

**Constraint Extraction System Prompt:**
```
You are a constraint extraction engine for a baby gift finder.
Extract: budget_aed, age_months, occasion, recipient, preferences, relationship, gender_preference, out_of_scope_reason
Respond ONLY with valid JSON. No markdown, no explanation.
```

**Recommendation Generation System Prompt:**
```
You are Mumzworld's AI Gift Finder. Recommend 3-5 baby gifts.
CRITICAL: Respond in SAME language as query. Generate Arabic NATIVELY — do NOT translate from English.
Confidence must be honest based on constraint matching.
```

## Tradeoffs

### Why This Problem?
I chose Gift Finder over the other options because:
1. **Highest grading criteria coverage**: Hits RAG, structured output + validation, multilingual, and agent design (constraint extraction)
2. **Demo-friendly**: Every reviewer immediately understands "gift for a 6-month-old"
3. **Uncertainty handling is natural**: Budget too low, age too old, vague queries → all have clear refusal paths
4. **Fits my existing skills**: RAG pipelines, structured outputs, and multilingual NLP

### Rejected Options
- **Return Reason Classifier**: Too simple, risks looking like "a single prompt wrapped in a script"
- **Product Image → PDP**: Sourcing clean product images without scraping is hard; less user-facing impact
- **Duplicate Product Detector**: No natural multilingual angle; harder to demo
- **Pediatric Symptom Triage**: Health-related (per your request to avoid); liability concerns

### Technical Tradeoffs
| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| ChromaDB local | Pinecone/Weaviate | Zero-config for reviewers; production would use managed vector DB |
| all-MiniLM-L6-v2 | OpenAI/Cohere embeddings | Free, local, sufficient for 165 products; would upgrade for 10K+ |
| LLaMA 3.3 70B | GPT-4/Claude | Free tier, strong Arabic, fast enough; would use GPT-4 for production |
| In-memory ChromaDB | Persistent ChromaDB | Simpler setup; data is small enough to reload on startup |
| No fine-tuning | Fine-tuned model | 5-hour constraint; few-shot prompting with retrieved context is sufficient |

### What I Cut
- **Image search**: Would require vision model, adds complexity, not core to gift finding
- **User history/personalization**: Out of scope for 5-hour prototype
- **Real-time inventory**: Using synthetic `in_stock` flag; production would integrate with ERP
- **A/B testing framework**: Mentioned in README as future work

### Known Failure Modes
1. **Arabic dialect variation**: Model trained on MSA; Gulf dialects may produce slightly formal Arabic
2. **Very niche requests**: "Organic bamboo teething toy under 50 AED" — may retrieve but with low confidence
3. **Ambiguous age references**: "1.5 years" vs "18 months" — handled by LLM extraction but edge cases exist
4. **Budget parsing**: "around 100" vs "under 100" — LLM sometimes extracts 100 for both; tolerance helps

## File Structure

```
mumzworld-gift-finder/
├── README.md              # This file
├── EVALS.md               # Detailed evaluation results
├── TRADEOFFS.md           # Architecture and product decisions
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
├── app.py                 # FastAPI backend serving HTML + /api/find
├── Landing_page.html      # Search page
├── Results_page.html      # Results page + transparency drawer
├── Dockerfile             # Free Hugging Face Spaces deployment
├── eval_runner.py         # Evaluation runner
├── data/
│   └── products.json      # 165 synthetic bilingual products
├── src/
│   ├── schema.py          # Pydantic output schemas
│   ├── catalog.py         # ChromaDB loader + retriever
│   ├── gift_finder.py     # Main LLM pipeline
│   └── language_utils.py  # Language detection + formatting
└── evals/
    ├── eval_cases.json    # 15 test cases
    └── eval_report.json   # Generated evaluation report
```

## What "Good" Looks Like (Per Brief)

- **Grounded in input**: Model returns null/refuses when answer not supported by catalog
- **Multilingual native copy**: Arabic reads like native copy, not translation (enforced by validators)
- **Structured output validates**: Pydantic schema with custom validators; failures are explicit
- **Evals before done**: 15 test cases covering easy, hard, edge, and adversarial inputs
- **Documentation first-class**: This README + inline comments + architecture explanation

## License

Built for Mumzworld AI Engineering Intern assessment. All code is original work with AI assistance documented.
