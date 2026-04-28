"""
Main Gift Finder pipeline.

Flow: language detection -> constraint extraction -> vector retrieval ->
hard filtering -> LLM re-ranking -> Pydantic validation. The important
submission detail is that uncertainty is handled in code, not only by prompt.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

try:
    from .catalog import search_products
    from .language_utils import detect_language, format_age, get_refusal_message
    from .schema import GiftFinderResponse, GiftRecommendation
except ImportError:
    from catalog import search_products
    from language_utils import detect_language, format_age, get_refusal_message
    from schema import GiftFinderResponse, GiftRecommendation

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MIN_BUDGET_AED = 35
MAX_AGE_MONTHS = 36
LOW_CONFIDENCE = 0.7
LOW_RETRIEVAL_SIMILARITY = 0.18


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env before running the app.")
    return Groq(api_key=api_key)


def _normalize_number(text: str) -> str:
    arabic_indic = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(arabic_indic)


def _empty_constraints() -> dict[str, Any]:
    return {
        "budget_aed": None,
        "age_months": None,
        "age_text": None,
        "occasion": None,
        "recipient": None,
        "preferences": [],
        "relationship": None,
        "gender_preference": None,
        "out_of_scope_reason": None,
    }


def _extract_constraints_locally(query: str) -> dict[str, Any]:
    """
    Lightweight deterministic extraction used both as a fallback and as a guardrail.
    It catches the eval-critical refusal cases even if the LLM extractor fails.
    """
    constraints = _empty_constraints()
    normalized = _normalize_number(query or "")
    lowered = normalized.lower()

    if not lowered.strip():
        return constraints

    if re.search(r"\b(dog|cat|pet|puppy|kitten)\b", lowered) or any(
        word in normalized for word in ["كلب", "قطة", "حيوان"]
    ):
        constraints["out_of_scope_reason"] = "The request is for a pet, not a baby or mom gift."

    budget_match = re.search(
        r"(?:under|below|less than|budget(?: of)?|up to|<)\s*(\d+(?:\.\d+)?)", lowered
    )
    if not budget_match:
        budget_match = re.search(r"(?:أقل من|تحت|ميزانية|حتى)\s*(\d+(?:\.\d+)?)", normalized)
    if not budget_match:
        budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:aed|درهم)", lowered)
    if budget_match:
        constraints["budget_aed"] = float(budget_match.group(1))

    age_match = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?(?:month|months|mo)\b", lowered)
    if age_match:
        constraints["age_months"] = int(round(float(age_match.group(1))))
        constraints["age_text"] = age_match.group(0)

    year_match = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?(?:year|years|yr)\b", lowered)
    if year_match:
        constraints["age_months"] = int(round(float(year_match.group(1)) * 12))
        constraints["age_text"] = year_match.group(0)

    arabic_month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:شهر|أشهر|اشهر)", normalized)
    if arabic_month_match:
        constraints["age_months"] = int(round(float(arabic_month_match.group(1))))
        constraints["age_text"] = arabic_month_match.group(0)

    arabic_year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:سنة|سنوات|عام)", normalized)
    if arabic_year_match:
        constraints["age_months"] = int(round(float(arabic_year_match.group(1)) * 12))
        constraints["age_text"] = arabic_year_match.group(0)

    if any(term in lowered for term in ["newborn", "new born"]):
        constraints["age_months"] = 0
        constraints["age_text"] = "newborn"
    if any(term in normalized for term in ["حديث الولادة", "مولود", "أنجبت", "انجبت"]):
        constraints["age_months"] = 0
        constraints["age_text"] = "newborn"
    if "سنة ونص" in normalized or "سنة ونصف" in normalized:
        constraints["age_months"] = 18
        constraints["age_text"] = "سنة ونص"

    if any(term in lowered for term in ["mom", "mother", "wife", "postpartum", "gave birth"]):
        constraints["recipient"] = "mom"
    if any(term in normalized for term in ["الأم", "ام", "أم", "زوجتي", "أختي", "اختي", "صديقتي"]):
        constraints["recipient"] = "mom"
    if any(term in lowered for term in ["baby", "newborn", "toddler", "child"]):
        constraints["recipient"] = constraints["recipient"] or "baby"
    if any(term in normalized for term in ["طفل", "طفلة", "رضيع", "بيبي"]):
        constraints["recipient"] = "baby"

    preference_map = {
        "educational": ["educational", "learning", "تعليمي", "تعليمية"],
        "practical": ["practical", "useful", "essentials", "عملي", "عملية"],
        "luxury": ["luxury", "premium", "fancy", "فاخر", "فاخرة"],
        "travel": ["travel", "stroller", "compact", "lightweight", "سفر", "عربة"],
        "teething": ["teething", "soothing", "cranky", "تسنين"],
        "organic": ["organic", "natural", "عضوي", "طبيعي"],
    }
    preferences = []
    for label, keywords in preference_map.items():
        if any(keyword in lowered or keyword in normalized for keyword in keywords):
            preferences.append(label)
    constraints["preferences"] = preferences

    if any(term in lowered for term in ["girl", "daughter"]) or any(
        term in normalized for term in ["بنت", "طفلة"]
    ):
        constraints["gender_preference"] = "girl"
    elif any(term in lowered for term in ["boy", "son"]) or any(
        term in normalized for term in ["ولد", "طفل"]
    ):
        constraints["gender_preference"] = "boy"

    return constraints


def _merge_constraints(local: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    merged = _empty_constraints()
    for key in merged:
        llm_value = llm.get(key)
        local_value = local.get(key)
        if key == "preferences":
            merged[key] = list(dict.fromkeys((local_value or []) + (llm_value or [])))
        else:
            merged[key] = local_value if local_value not in (None, "", []) else llm_value

    if local.get("out_of_scope_reason"):
        merged["out_of_scope_reason"] = local["out_of_scope_reason"]
    return merged


def extract_constraints(query: str, language: str) -> dict[str, Any]:
    """
    Extract structured constraints from natural language.

    The LLM gives richer understanding; local rules protect the critical edge
    cases: empty queries, pets, low budgets, and ages outside 0-36 months.
    """
    local_constraints = _extract_constraints_locally(query)
    if not query or not query.strip():
        return local_constraints

    system_prompt = """You are a constraint extraction engine for a baby and mom gift finder.
Extract these fields from the user's query:
- budget_aed: maximum budget in AED as a number, or null
- age_months: baby age in months as a number, or null
- age_text: original age phrase, or null
- occasion: baby_shower, birthday, eid, newborn_visit, just_because, or null
- recipient: baby, mom, dad, parents, or null
- preferences: array of short labels such as practical, luxury, educational, organic, travel, teething
- relationship: friend, sister, colleague, wife, self, or null
- gender_preference: boy, girl, neutral, or null
- out_of_scope_reason: short reason if clearly not baby/mom/parent related, otherwise null

Important:
- "1.5 years", "one and a half", and Arabic "سنة ونص" mean 18 months.
- New mom, postpartum, wife after birth, sister who gave birth are in scope.
- Pet, adult-only, electronics, or unrelated requests are out of scope.

Respond ONLY with valid JSON. No markdown."""

    try:
        response = _get_groq_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query ({language}): {query}"},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        llm_constraints = json.loads(response.choices[0].message.content)
    except Exception as exc:
        print(f"Constraint extraction fell back to local rules: {exc}")
        llm_constraints = {}

    return _merge_constraints(local_constraints, llm_constraints)


def filter_candidates(candidates: list[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply hard business constraints after retrieval."""
    filtered = []
    budget = constraints.get("budget_aed")
    age = constraints.get("age_months")
    recipient = constraints.get("recipient")

    for candidate in candidates:
        if budget is not None and candidate["price_aed"] > float(budget) * 1.1:
            continue

        if age is not None and not (
            candidate["age_min"] - 2 <= int(age) <= candidate["age_max"] + 2
        ):
            if recipient != "mom":
                continue

        if recipient == "mom" and candidate["category_en"] not in [
            "Postpartum",
            "Feeding",
            "Health & Safety",
            "Sleep",
        ]:
            continue

        if not candidate.get("in_stock", True):
            continue

        filtered.append(candidate)

    return filtered


def _make_refusal(
    query: str,
    language: str,
    reason: str,
    constraints: dict[str, Any],
    understood_en: str | None = None,
    understood_ar: str | None = None,
) -> GiftFinderResponse:
    refusal = get_refusal_message(language, reason)
    return GiftFinderResponse(
        query_understood_en=understood_en or f"I could not confidently answer this request: {query}",
        query_understood_ar=understood_ar or f"لا أستطيع الإجابة بثقة على هذا الطلب: {query}",
        recommendations=[],
        out_of_scope=True,
        uncertainty_note_en=refusal["message_en"],
        uncertainty_note_ar=refusal["message_ar"],
        language_detected=language,
        budget_extracted=constraints.get("budget_aed"),
        age_months_extracted=constraints.get("age_months"),
    )


def _vague_request(query: str, constraints: dict[str, Any]) -> bool:
    return bool(query.strip()) and not any(
        [
            constraints.get("budget_aed"),
            constraints.get("age_months") is not None,
            constraints.get("preferences"),
            constraints.get("recipient") == "mom",
        ]
    )


def _candidate_context(candidates: list[dict[str, Any]]) -> str:
    blocks = []
    for index, candidate in enumerate(candidates[:8], start=1):
        blocks.append(
            "\n".join(
                [
                    f"Product {index}:",
                    f"ID: {candidate['id']}",
                    f"Name (EN): {candidate['name_en']}",
                    f"Name (AR): {candidate['name_ar']}",
                    f"Category (EN): {candidate['category_en']}",
                    f"Category (AR): {candidate['category_ar']}",
                    f"Price: {candidate['price_aed']} AED",
                    f"Age: {candidate['age_min']}-{candidate['age_max']} months",
                    f"Rating: {candidate['avg_rating']}/5 ({candidate['num_reviews']} reviews)",
                    f"In Stock: {candidate['in_stock']}",
                    f"Retrieval Similarity: {candidate.get('similarity', 0):.2f}",
                    f"Tags: {', '.join(candidate['tags'])}",
                    f"Brand: {candidate['brand']}",
                    f"Description (EN): {candidate['description_en']}",
                    f"Description (AR): {candidate['description_ar']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _grounded_reason_en(candidate: dict[str, Any], constraints: dict[str, Any]) -> str:
    """Build the final English reason only from catalog fields."""
    tags = ", ".join(candidate["tags"][:4]) if candidate.get("tags") else "catalog-matched"
    budget = constraints.get("budget_aed")
    budget_phrase = (
        f" It is within the extracted {budget:.0f} AED budget."
        if budget is not None and candidate["price_aed"] <= float(budget) * 1.1
        else ""
    )
    return (
        "This recommendation is grounded in the product catalog: "
        f"{candidate['name_en']} is a {candidate['category_en']} item priced at "
        f"{candidate['price_aed']} AED, suitable for ages {candidate['age_min']}-"
        f"{candidate['age_max']} months, tagged with {tags}, rated "
        f"{candidate['avg_rating']}/5 from {candidate['num_reviews']} reviews, "
        f"and marked {'in stock' if candidate['in_stock'] else 'out of stock'}."
        f"{budget_phrase}"
    )


def _grounded_reason_ar(candidate: dict[str, Any], constraints: dict[str, Any]) -> str:
    """Build the final Arabic reason only from catalog fields."""
    budget = constraints.get("budget_aed")
    budget_phrase = (
        f" كما أنه ضمن الميزانية المستخرجة وهي {budget:.0f} درهم."
        if budget is not None and candidate["price_aed"] <= float(budget) * 1.1
        else ""
    )
    stock_text = "متوفر حالياً" if candidate["in_stock"] else "غير متوفر حالياً"
    return (
        "هذا الترشيح مبني على بيانات الكتالوج فقط: "
        f"{candidate['name_ar']} من فئة {candidate['category_ar']}، وسعره "
        f"{candidate['price_aed']} درهم، ومناسب لعمر من {candidate['age_min']} إلى "
        f"{candidate['age_max']} شهر، وتقييمه {candidate['avg_rating']} من 5 بناءً على "
        f"{candidate['num_reviews']} مراجعة، وهو {stock_text}."
        f"{budget_phrase}"
    )


def _ground_response_in_candidates(
    response: GiftFinderResponse,
    candidates: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> GiftFinderResponse:
    """
    Enforce grounding after generation.

    The LLM may rank and select products, but final product details and reasons
    are overwritten from retrieved catalog records to prevent hallucinated
    names, prices, stock status, age ranges, or unsupported features.
    """
    if response.out_of_scope:
        return response

    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    grounded_recommendations: list[GiftRecommendation] = []
    seen_ids: set[str] = set()

    for recommendation in response.recommendations:
        candidate = candidate_by_id.get(recommendation.product_id)
        if not candidate or candidate["id"] in seen_ids:
            continue

        seen_ids.add(candidate["id"])
        confidence = min(
            recommendation.confidence,
            max(0.45, min(0.95, 0.55 + candidate.get("similarity", 0.0))),
        )

        grounded_recommendations.append(
            GiftRecommendation(
                product_id=candidate["id"],
                product_name_en=candidate["name_en"],
                product_name_ar=candidate["name_ar"],
                price_aed=candidate["price_aed"],
                category_en=candidate["category_en"],
                category_ar=candidate["category_ar"],
                reason_en=_grounded_reason_en(candidate, constraints),
                reason_ar=_grounded_reason_ar(candidate, constraints),
                confidence=confidence,
                age_suitability=format_age(candidate["age_min"], candidate["age_max"], "en"),
                in_stock=candidate["in_stock"],
            )
        )

    response.recommendations = grounded_recommendations
    if not grounded_recommendations:
        response.out_of_scope = True
        response.uncertainty_note_en = (
            "I do not know which product to recommend because the model did not choose "
            "a product ID from the retrieved catalog evidence."
        )
        response.uncertainty_note_ar = (
            "لا أعرف أي منتج أوصي به لأن النموذج لم يختر معرف منتج من نتائج الكتالوج المسترجعة."
        )

    return GiftFinderResponse(**response.model_dump())


def _fallback_recommendations(
    query: str,
    language: str,
    candidates: list[dict[str, Any]],
    constraints: dict[str, Any],
    note_en: str,
    note_ar: str,
) -> GiftFinderResponse:
    recommendations = []
    for candidate in candidates[:3]:
        confidence = max(0.45, min(0.72, candidate.get("similarity", 0.5)))
        recommendations.append(
            GiftRecommendation(
                product_id=candidate["id"],
                product_name_en=candidate["name_en"],
                product_name_ar=candidate["name_ar"],
                price_aed=candidate["price_aed"],
                category_en=candidate["category_en"],
                category_ar=candidate["category_ar"],
                reason_en=_grounded_reason_en(candidate, constraints),
                reason_ar=_grounded_reason_ar(candidate, constraints),
                confidence=confidence,
                age_suitability=format_age(candidate["age_min"], candidate["age_max"], "en"),
                in_stock=candidate["in_stock"],
            )
        )

    return GiftFinderResponse(
        query_understood_en=f"Cautious fallback recommendations for: {query}",
        query_understood_ar=f"ترشيحات احتياطية بحذر لطلب: {query}",
        recommendations=recommendations,
        out_of_scope=False,
        uncertainty_note_en=note_en,
        uncertainty_note_ar=note_ar,
        language_detected=language,
        budget_extracted=constraints.get("budget_aed"),
        age_months_extracted=constraints.get("age_months"),
    )


def _enforce_uncertainty(
    response: GiftFinderResponse,
    language: str,
    candidates: list[dict[str, Any]],
    constraints: dict[str, Any],
    query: str,
) -> GiftFinderResponse:
    if response.out_of_scope:
        if not response.uncertainty_note_en or not response.uncertainty_note_ar:
            refusal = get_refusal_message(language, "out_of_scope")
            response.uncertainty_note_en = refusal["message_en"]
            response.uncertainty_note_ar = refusal["message_ar"]
        return response

    top_similarity = max((c.get("similarity", 0.0) for c in candidates), default=0.0)
    lowest_llm_confidence = min((rec.confidence for rec in response.recommendations), default=0.0)
    vague = _vague_request(query, constraints)

    if vague:
        response.uncertainty_note_en = (
            "I found a few generally suitable gifts, but I am uncertain because the request "
            "does not include an age, budget, or clear preference."
        )
        response.uncertainty_note_ar = (
            "وجدت بعض الهدايا المناسبة بشكل عام، لكنني غير متأكد لأن الطلب لا يذكر العمر أو الميزانية أو تفضيلاً واضحاً."
        )
        for rec in response.recommendations:
            rec.confidence = min(rec.confidence, 0.68)
    elif top_similarity < LOW_RETRIEVAL_SIMILARITY or lowest_llm_confidence < LOW_CONFIDENCE:
        response.uncertainty_note_en = (
            "I am not fully certain about this match because the catalog evidence is weak or "
            "one of the requested constraints is only partially covered."
        )
        response.uncertainty_note_ar = (
            "لست متأكداً تماماً من هذا الترشيح لأن دليل الكتالوج محدود أو لأن أحد الشروط مغطى جزئياً فقط."
        )
        for rec in response.recommendations:
            rec.confidence = min(rec.confidence, 0.69)

    response.language_detected = language
    response.budget_extracted = constraints.get("budget_aed")
    response.age_months_extracted = constraints.get("age_months")
    return GiftFinderResponse(**response.model_dump())


def generate_recommendations(
    query: str,
    language: str,
    candidates: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> GiftFinderResponse:
    """Generate final recommendations using the retrieved product evidence."""
    budget = constraints.get("budget_aed")
    age = constraints.get("age_months")

    if not query or not query.strip():
        return _make_refusal(query, language, "empty_query", constraints, "Empty query received", "تم استلام طلب فارغ")

    if constraints.get("out_of_scope_reason"):
        return _make_refusal(
            query,
            language,
            "not_baby_related",
            constraints,
            f"Out-of-scope request: {constraints['out_of_scope_reason']}",
            f"طلب خارج النطاق: {constraints['out_of_scope_reason']}",
        )

    if budget is not None and float(budget) < MIN_BUDGET_AED:
        return _make_refusal(
            query,
            language,
            "budget_too_low",
            constraints,
            f"Budget too low: {budget} AED",
            f"الميزانية منخفضة جداً: {budget} درهم",
        )

    if age is not None and int(age) > MAX_AGE_MONTHS:
        return _make_refusal(
            query,
            language,
            "age_too_old",
            constraints,
            f"Age {age} months exceeds the 0-36 month prototype scope.",
            f"العمر {age} شهر يتجاوز نطاق النموذج الأولي من 0 إلى 36 شهر.",
        )

    if not candidates:
        return _make_refusal(
            query,
            language,
            "out_of_scope",
            constraints,
            f"I do not know which catalog item fits this request: {query}",
            f"لا أعرف أي منتج في الكتالوج يناسب هذا الطلب: {query}",
        )

    system_prompt = f"""You are Mumzworld's AI Gift Finder for baby and mom gifts in the UAE.
This is a two-step RAG system:
- Retrieval has already found possible catalog products.
- Generation may only choose from and explain those retrieved products.

Recommend 3 to 5 products using ONLY the retrieved catalog items.

Critical rules:
1. Use only product IDs from the retrieved candidates.
2. If the candidates do not support the request, set out_of_scope=true and explain uncertainty.
3. Confidence must be honest. Lower confidence if age, budget, recipient, or preference is only a partial match.
4. Arabic fields must be natural Modern Standard Arabic. Do not transliterate.
5. Reasons must mention exact evidence: price, age range, category, tags, stock, or rating.
6. Include `language_detected` exactly as "{language}".
7. Never invent product features, materials, colors, benefits, bundles, or medical claims.
8. If you are not sure, say you do not know or need more information.

Return strict JSON matching this shape:
{{
  "query_understood_en": "English interpretation",
  "query_understood_ar": "Arabic interpretation",
  "recommendations": [
    {{
      "product_id": "P001",
      "product_name_en": "English catalog name",
      "product_name_ar": "Arabic catalog name",
      "price_aed": 149.0,
      "category_en": "Category",
      "category_ar": "الفئة",
      "reason_en": "Specific reason with catalog evidence, at least 20 words.",
      "reason_ar": "سبب عربي محدد يستند إلى بيانات الكتالوج ولا يقل عن عشرين كلمة.",
      "confidence": 0.82,
      "age_suitability": "6-12 months",
      "in_stock": true
    }}
  ],
  "out_of_scope": false,
  "uncertainty_note_en": null,
  "uncertainty_note_ar": null,
  "language_detected": "{language}",
  "budget_extracted": {json.dumps(budget)},
  "age_months_extracted": {json.dumps(age)}
}}"""

    user_prompt = (
        f"User Query ({language}): {query}\n\n"
        f"Extracted Constraints: {json.dumps(constraints, ensure_ascii=False)}\n\n"
        f"Retrieved Candidates:\n{_candidate_context(candidates)}"
    )

    try:
        response = _get_groq_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        response_obj = GiftFinderResponse(**parsed)
        response_obj = _ground_response_in_candidates(response_obj, candidates, constraints)
        return _enforce_uncertainty(response_obj, language, candidates, constraints, query)
    except Exception as exc:
        print(f"Recommendation generation fell back to cautious mode: {exc}")
        return _fallback_recommendations(
            query=query,
            language=language,
            candidates=candidates,
            constraints=constraints,
            note_en=(
                "I am uncertain because the LLM response could not be validated. "
                "These recommendations come directly from the retrieval layer."
            ),
            note_ar=(
                "أنا غير متأكد لأن استجابة النموذج لم تجتز التحقق. هذه الترشيحات مأخوذة مباشرة من طبقة البحث."
            ),
        )


def find_gifts(query: str) -> GiftFinderResponse:
    """Main entry point for the UI, API, and eval runner."""
    language = detect_language(query)
    constraints = extract_constraints(query, language)

    if (
        not query
        or not query.strip()
        or constraints.get("out_of_scope_reason")
        or (
            constraints.get("budget_aed") is not None
            and float(constraints["budget_aed"]) < MIN_BUDGET_AED
        )
        or (
            constraints.get("age_months") is not None
            and int(constraints["age_months"]) > MAX_AGE_MONTHS
        )
    ):
        return generate_recommendations(query, language, [], constraints)

    raw_candidates = search_products(query, n_results=15)
    filtered_candidates = filter_candidates(raw_candidates, constraints)

    return generate_recommendations(query, language, filtered_candidates, constraints)
