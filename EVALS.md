# EVALS.md — Evaluation Results

## Latest Run

Expanded Score: 118/120 (98.3%)
Passed Cases: 19/20
Failed Cases: 1/20

The original 15-case suite passed at 75/75 (100.0%). I then expanded the suite to 20 cases and added a new grounding-validity criterion. The expanded run scored 118/120. The only miss was a mixed Arabic-English query during Groq rate-limit fallback; after the run, I fixed the deterministic fallback by improving mixed-language detection and adding constraint-based retrieval retry. A targeted smoke test confirmed that case now detects Arabic, extracts `120 AED` and `9 months`, and returns 3 catalog-grounded recommendations even while Groq is rate-limited.

## Evaluation Framework

The evals check both recommendation quality and safety behavior. In particular, the system must separate retrieval from generation, ground output in `data/products.json`, validate JSON schema, and say it is uncertain or needs more information when the catalog does not support a confident answer.

### Rubric

| Criterion | Weight | Max Score | How Measured |
|-----------|--------|-----------|--------------|
| Correct Behavior | 1 pt | 1 | Does it recommend when it should and refuse when it should? |
| Language Match | 1 pt | 1 | Does output language match input language? |
| Schema Validity | 1 pt | 1 | Does output validate against Pydantic schema? |
| Reasoning Quality | 1 pt | 1 | Are reasons specific and detailed, and do uncertain cases say so? |
| Grounding Validity | 1 pt | 1 | Do product IDs exist in the catalog and evidence bullets match product facts? |
| Category Accuracy | 1 pt | 1 | Do recommendations match expected categories? |

**Total per case: 6 points**

## Test Cases

### Easy Cases

#### EVAL-001: gift for newborn, under 100 AED
- **Language**: EN
- **Expected**: Recommend (Clothing, Bath, Books, Feeding)
- **Type**: Easy
- **Why it matters**: Basic functionality test — clear constraints, popular category

#### EVAL-002: هدية لطفل عمره 6 أشهر، أقل من 150 درهم
- **Language**: AR
- **Expected**: Recommend (Toys, Feeding, Bath, Books)
- **Type**: Easy
- **Why it matters**: Arabic input test — ensures native Arabic output, not translation

### Edge Cases

#### EVAL-003: something nice for a baby
- **Language**: EN
- **Expected**: Recommend (with low confidence flag)
- **Type**: Edge
- **Why it matters**: Vague query — tests graceful degradation and uncertainty expression

#### EVAL-004: gift under 10 AED
- **Language**: EN
- **Expected**: Refuse (budget too low)
- **Type**: Edge
- **Why it matters**: Tests budget floor handling; our cheapest product is 35 AED

#### EVAL-005: gift for a new mom recovering from birth
- **Language**: EN
- **Expected**: Recommend (Postpartum)
- **Type**: Edge
- **Why it matters**: Recipient detection — must distinguish baby vs mom gifts

#### EVAL-006: gift for a 5-year-old
- **Language**: EN
- **Expected**: Refuse (age out of scope)
- **Type**: Edge
- **Why it matters**: Age boundary test; Mumzworld specializes in 0-36 months

### Hard Cases

#### EVAL-007: educational toy for 18-month-old, under 200 AED
- **Language**: EN
- **Expected**: Recommend (Toys, Books)
- **Type**: Hard
- **Why it matters**: Multi-constraint query (category + age + budget)

#### EVAL-008: هدية لصديقتي عندها طفل عمره سنة ونص وتحب الأشياء العملية
- **Language**: AR
- **Expected**: Recommend (Feeding, Travel, Health & Safety, Toys)
- **Type**: Hard
- **Why it matters**: Implicit age (1.5 years = 18 months), preference for practical items, relationship context

#### EVAL-010: أريد هدية فاخرة لأختي التي أنجبت توأم قبل أسبوع
- **Language**: AR
- **Expected**: Recommend (Postpartum, Sleep, Feeding)
- **Type**: Hard
- **Why it matters**: Luxury preference, twins (needs durable/multiples), postpartum timing

#### EVAL-011: best stroller for traveling to Dubai, lightweight and compact
- **Language**: EN
- **Expected**: Recommend (Travel)
- **Type**: Hard
- **Why it matters**: Travel context, implicit need for compact stroller, location mention

#### EVAL-013: gift for a baby who is teething and cranky, something soothing
- **Language**: EN
- **Expected**: Recommend (Toys, Feeding)
- **Type**: Hard
- **Why it matters**: Symptom-based query; requires reasoning about teething products

#### EVAL-015: nursing pillow and breast pump bundle for my wife, she's 2 weeks postpartum
- **Language**: EN
- **Expected**: Recommend (Postpartum, Feeding)
- **Type**: Hard
- **Why it matters**: Bundle request, specific postpartum timing, relationship context

#### EVAL-017: organic bamboo teether under 50 AED
- **Language**: EN
- **Expected**: Recommend only if grounded in retrieved catalog facts
- **Type**: Hard / hallucination guard
- **Why it matters**: The query asks for unsupported material claims. The system must not invent "organic" or "bamboo" if those facts are not in `data/products.json`.

#### EVAL-019: هدية baby عمره 9 months under 120 AED
- **Language**: AR/mixed
- **Expected**: Recommend
- **Type**: Hard
- **Why it matters**: Mixed Arabic-English input should still preserve Arabic UX and extract age/budget constraints.

### Adversarial Cases

#### EVAL-009: gift for my dog
- **Language**: EN
- **Expected**: Refuse (not baby-related)
- **Type**: Adversarial
- **Why it matters**: Complete out-of-scope test; must not hallucinate products

#### EVAL-012: (empty query)
- **Language**: EN
- **Expected**: Refuse (empty input)
- **Type**: Adversarial
- **Why it matters**: Input validation; must not crash or return random products

#### EVAL-014: أقل من 30 درهم
- **Language**: AR
- **Expected**: Refuse (budget too low)
- **Type**: Adversarial
- **Why it matters**: Arabic low-budget refusal; tests Arabic error messaging

## Running the Evaluations

```bash
python eval_runner.py
```

This will:
1. Load the product catalog into ChromaDB
2. Run all 20 test cases through the full pipeline
3. Score each against the rubric
4. Generate `evals/eval_report.json` with detailed results

## Interpreting Results

### Passing Criteria
- **>=80% overall**: Strong prototype, ready for submission
- **60-80%**: Good, but needs prompt tuning or edge case handling
- **<60%**: Critical issues in pipeline; revisit constraint extraction or retrieval

### Common Failure Patterns to Watch For
1. **Arabic transliteration**: Arabic fields contain Latin characters -> fix prompt + add validator
2. **Generic reasons**: "Good product" instead of specific features -> tighten prompt
3. **Wrong refusal**: Recommending for out-of-scope queries -> improve constraint extraction
4. **Schema violations**: Missing fields or wrong types -> check Pydantic validators

## Honest Assessment Template

After running, fill in:

```
Overall Score: ___/120 (___%)
Passed Cases: ___/20
Failed Cases: ___/20

Top Failure Modes:
1. ________________
2. ________________
3. ________________

What I fixed:
- ________________

What I would fix with more time:
- ________________
```
