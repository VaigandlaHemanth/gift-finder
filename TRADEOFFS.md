# TRADEOFFS.md — Architecture & Product Decisions

## Problem Selection

### Why Gift Finder?

I evaluated all Track A options against four criteria: grading rubric coverage, demo impact, build feasibility in 5 hours, and alignment with my skills.

| Option | RAG | Multimodal | Structured Output | Agent Design | Multilingual | Demo Impact | 5hr Feasible | My Fit |
|--------|-----|-----------|-------------------|--------------|--------------|-------------|--------------|--------|
| Gift Finder | Yes | No | Yes | Yes | Yes | High | Yes | High |
| Product Image -> PDP | No | Yes | Yes | No | Yes | Medium | Maybe | Medium |
| Return Classifier | No | No | Yes | No | Yes | Low | Yes | Medium |
| Reviews -> Moms Verdict | Yes | No | Yes | No | Yes | High | Maybe | Medium |
| Email Triage | No | No | Yes | No | Yes | Medium | Yes | Medium |
| Duplicate Detector | Yes | No | Yes | No | No | Low | Yes | Medium |
| Ops Dashboard | No | No | Yes | No | No | Medium | Maybe | Low |
| Pregnancy Timeline | No | No | Yes | Yes | Yes | High | Maybe | Medium |
| Pediatric Triage | No | No | Yes | Yes | Yes | High | Maybe | Low |

**Gift Finder won because:**
1. Hits 4/5 grading criteria (only misses multimodal, which is optional)
2. Every reviewer immediately grasps the value proposition
3. Natural uncertainty handling paths (budget, age, scope)
4. Leverages my existing RAG and structured output experience

### Why Not Health-Related?

I explicitly avoided the Pediatric Symptom Triage option because:
- **Liability**: Health recommendations carry medical liability; a prototype should not risk user safety
- **Evaluation difficulty**: Hard to validate "correct" triage without medical expertise
- **Brief guidance**: The examples list it as one option among many; non-health alternatives are equally valid

## Model Selection

### LLM: Groq LLaMA 3.3 70B vs Alternatives

| Model | Cost | Arabic | Speed | Context | Why/Why Not |
|-------|------|--------|-------|---------|-------------|
| **LLaMA 3.3 70B** | Free | Strong | Very Fast | 128K | Best free-tier Arabic, fast enough |
| GPT-4o | Paid | Excellent | Fast | 128K | Requires paid API, breaks "all free" constraint |
| Claude 3.5 Sonnet | Paid | Excellent | Fast | 200K | Paid only, though excellent at Arabic |
| Qwen 2.5 72B | Free | Strong | Fast | 128K | Strong Arabic but slightly slower on Groq |
| Gemma 2 9B | Free | Weak | Fastest | 8K | Arabic too weak for native copy generation |

**Decision**: LLaMA 3.3 70B on Groq. Free tier is generous (20 requests/min, 1M tokens/day), Arabic is strong enough for native copy, and inference is fast (<2s per call).

### Embeddings: Local vs API

| Option | Cost | Setup | Quality | Decision |
|--------|------|-------|---------|----------|
| all-MiniLM-L6-v2 (local) | Free | pip install | Good for 165 products | Chosen |
| Cohere embed-english-v3 | Free tier | API key | Better | Extra dependency |
| OpenAI text-embedding-3 | Paid | API key | Best | Paid |
| multilingual-e5-large (local) | Free | pip install | Better multilingual | 1GB download, overkill |

**Decision**: all-MiniLM-L6-v2. At 165 products, the quality difference is negligible vs. larger models. Zero API keys for reviewers.

## Vector Store Selection

### ChromaDB vs Pinecone

| Factor | ChromaDB Local | Pinecone Free |
|--------|---------------|---------------|
| Setup time | 0 min | 10 min (signup + index creation) |
| API keys needed | 0 | 1 |
| Persistence | Local persistent directory, rebuilt when needed | Cloud persistent |
| Scale | <10K items comfortably | Millions |
| Query speed | Fast locally | Fast via network |

**Decision**: ChromaDB local. The brief emphasizes "runs in under 5 minutes from clone." Adding Pinecone means reviewers need to sign up, create an index, and configure an API key — violating that constraint. Production would absolutely use Pinecone or Weaviate.

## UI and Hosting

I used the custom `Landing_page.html` and `Results_page.html` as the actual interface instead of Streamlit. FastAPI serves both pages and exposes `/api/find`, which keeps the Groq API key on the server and lets the same repo deploy as a free Docker app on Hugging Face Spaces. The backend keeps bilingual structured fields for validation, while the visible UI renders only the detected user language.

The UI includes an improvement panel that can collect multiple missing details at once. This keeps the "need more information" behavior useful without interrupting the user with browser popups. Catalog prices are stored in AED; if a user asks in INR, the prototype displays approximate INR prices with a fixed conversion rate to avoid adding another external API key.

## Language and Currency Detection

The system detects Arabic with a Unicode heuristic plus `langdetect`, then renders the visible UI in only that language. English queries show English cards, reasons, refusals, and suggestions. Arabic queries show Arabic cards, reasons, refusals, and suggestions. Bilingual fields remain in the raw JSON because they are useful for validation and reviewer transparency.

Currency detection is deterministic. AED, درهم, or no explicit currency uses AED. INR, ₹, rupees, or Rs converts the extracted budget into AED for catalog filtering, then displays approximate INR prices in the UI. This is a prototype convenience for non-GCC reviewers; the catalog source of truth remains AED.

## Schema Design

### Why Pydantic v2?

Pydantic v2 provides:
1. **Runtime validation**: Catches malformed LLM outputs before they reach the user
2. **Custom validators**: We enforce Arabic quality (no Latin characters) and reason specificity
3. **JSON Schema generation**: Easy to document and test against
4. **Type safety**: Catches bugs at development time

### Custom Validators

```python
# Rejects generic reasons like "good product"
@field_validator('reason_en')
def reason_not_generic_en(cls, v):
    ...

# Rejects transliterated Arabic (Latin chars in Arabic field)
@field_validator('reason_ar')
def reason_not_generic_ar(cls, v):
    ...
```

These validators catch the #1 and #2 failure modes in multilingual LLM outputs.

## Arabic Handling Strategy

### The Translation Trap

Most LLMs, when asked to "respond in Arabic," will:
1. Generate English reasoning internally
2. Translate to Arabic at the output layer
3. Produce awkward, formal Arabic that feels robotic

### Our Solution

1. **Native generation prompt**: "Generate Arabic text NATIVELY — do NOT translate from English"
2. **Pydantic validator**: Rejects any Arabic field containing Latin characters
3. **Cultural context in prompts**: Mention GCC context (AED currency, regional preferences)
4. **Fallback handling**: If Arabic generation fails, lower confidence score rather than return bad Arabic

### Validation Without Knowing Arabic

Since I don't read Arabic, I used:
- **Unicode range checks**: Ensure Arabic fields contain Arabic script
- **Length checks**: Ensure Arabic reasons are substantial (>20 chars)
- **Pattern detection**: Reject obvious transliteration (mixed scripts)
- **Round-trip test**: Feed Arabic output back to LLM for quality assessment

## Uncertainty Handling Design

### Retrieval vs Generation Boundary

Retrieval is responsible for finding candidate products from `data/products.json`. Generation is responsible for ranking those candidates and explaining the match. To keep the boundary honest, the final response is grounded after the LLM call: product IDs must exist in the retrieved set, and user-facing product facts are overwritten from catalog data before schema validation.

### Refusal Triggers

| Trigger | Detection | Response |
|---------|-----------|----------|
| Budget < 35 AED | Constraint extraction | Refuse with budget guidance |
| Age > 36 months | Constraint extraction | Refuse with age guidance |
| Empty query | Input validation | Refuse with help message |
| Not baby-related | LLM out_of_scope_reason | Refuse with scope clarification |
| No candidates after filter | Retrieval + filtering | Refuse with suggestion to broaden |
| Low confidence (<0.7) | LLM/self-assessment + retrieval score | Recommend with uncertainty note and refinement suggestions |

### Evidence-Grounded Output

I added evidence bullets to each recommendation after the first prototype was working. This is intentionally deterministic: product ID, price, category, age range, rating/review count, stock, tags, and retrieval similarity are copied from the retrieved catalog record rather than generated by the LLM. The model can help rank, but it cannot invent product facts. I keep that evidence in the validated JSON/transparency layer rather than showing it as a large box on every product card, because the main UI should feel like shopping help, not an audit log.

### Why This Matters

The grading rubric weights uncertainty handling at 15%. A system that confidently recommends "dog toys" when asked for baby gifts fails immediately. Our multi-layer detection (constraint extraction -> filtering -> LLM assessment -> Pydantic validation) makes incorrect recommendations statistically unlikely.

## What Was Cut

### Cut: Image Search
- **Why**: Would require vision model (GPT-4V/Claude 3), adds complexity, not core to text-based gift finding
- **Impact**: Low — text queries are the primary input modality
- **Would add if**: 10+ hours, or if brief required multimodal

### Cut: User History
- **Why**: Requires user authentication, database, session management
- **Impact**: Medium — personalization would improve recommendations
- **Would add if**: Production deployment with user accounts

### Cut: Real Inventory Integration
- **Why**: No access to live ERP inventory; using synthetic in_stock flags
- **Impact**: Low for prototype; critical for production
- **Would add if**: API access to inventory system

### Cut: A/B Testing Framework
- **Why**: Out of scope for 5-hour prototype
- **Impact**: Low for assessment; critical for production iteration
- **Would add if**: Post-internship feature development

## Time Log

| Phase | Time | What |
|-------|------|------|
| Architecture & planning | 30 min | Problem selection, stack decisions, schema design |
| Dataset generation | 20 min | 165 synthetic bilingual products |
| Core pipeline | 90 min | catalog.py, retriever, gift_finder.py |
| Schema & validation | 30 min | Pydantic models, custom validators |
| UI development | 45 min | HTML results flow with FastAPI backend |
| Arabic prompt tuning | 30 min | Iterating prompts for native Arabic output |
| Evaluation | 60 min | 20 test cases, eval runner, grounding-validity scoring |
| Documentation | 30 min | README, EVALS.md, TRADEOFFS.md |
| Loom recording | 15 min | 5 inputs including 1 refusal |
| Hosting and polish | 45 min | Hugging Face Space, GitHub snapshot, transparency and suggestion UI |
| **Total** | **~6 hours** | Slightly over 5 hours to add hosted demo, same-language UI, and stronger eval polish |

## Production Path

If this were going to production:

1. **Swap ChromaDB for Pinecone/Weaviate** for persistent, scalable vector search
2. **Add fine-tuned embedding model** on the real product corpus for better retrieval
3. **Integrate real inventory API** for live stock levels and pricing
4. **Add user session history** for personalized recommendations
5. **Implement A/B testing** with conversion rate as North Star metric
6. **Add image multimodality** for "find gifts like this photo" feature
7. **Expand to 10K+ products** with category-specific embedding spaces
8. **Add human-in-the-loop feedback** for continuous improvement
