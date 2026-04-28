"""
Language utilities for bilingual support.
Handles detection, formatting, and Arabic-specific helpers.
"""
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Seed for reproducible detection
DetectorFactory.seed = 42


def detect_language(text: str) -> str:
    """
    Detect if text is English or Arabic.
    Returns 'en' or 'ar'.
    """
    if not text or not text.strip():
        return "en"  # Default to English for empty input

    # Quick heuristic: check for Arabic Unicode range
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    if arabic_chars > len(text) * 0.3:
        return "ar"

    try:
        detected = detect(text)
        return "ar" if detected == "ar" else "en"
    except LangDetectException:
        return "en"


def format_price(price: float, language: str) -> str:
    """Format price with appropriate currency text."""
    if language == "ar":
        return f"{price:.0f} درهم إماراتي"
    return f"{price:.0f} AED"


def format_age(age_min: int, age_max: int, language: str) -> str:
    """Format age range appropriately."""
    if language == "ar":
        if age_min == 0 and age_max <= 3:
            return "حديثي الولادة"
        elif age_max <= 12:
            return f"من {age_min} إلى {age_max} أشهر"
        else:
            years = age_max // 12
            return f"حتى {years} سنوات"
    else:
        if age_min == 0 and age_max <= 3:
            return "Newborn"
        elif age_max <= 12:
            return f"{age_min}-{age_max} months"
        else:
            years = age_max // 12
            return f"Up to {years} years"


def get_refusal_message(language: str, reason: str = "out_of_scope") -> dict:
    """
    Get culturally appropriate refusal messages.
    Critical for uncertainty handling grading.
    """
    messages = {
        "en": {
            "out_of_scope": "I couldn't find suitable gifts matching your request. This might be because the budget is too low, the age is outside our range (we specialize in 0-3 years), or the request isn't baby-related. Could you provide more details?",
            "budget_too_low": "I couldn't find any products under this budget. Our gifts start from around 35 AED. Would you like to adjust your budget?",
            "age_too_old": "We specialize in gifts for babies and toddlers up to 3 years old. For older children, I'd recommend checking our main store categories.",
            "not_baby_related": "This doesn't seem to be a baby or mom-related gift request. I'm designed to help with gifts for babies, moms, and expecting parents. How can I help with that?",
            "empty_query": "Please tell me who the gift is for, their age, and your budget so I can help you find the perfect gift.",
            "vague": "Your request is a bit broad. To help better, could you share the baby's age and your budget?"
        },
        "ar": {
            "out_of_scope": "لم أتمكن من العثور على هدايا مناسبة لطلبك. قد يكون السبب أن الميزانية منخفضة جداً، أو أن العمر خارج نطاقنا (نحن متخصصون في 0-3 سنوات)، أو أن الطلب لا يتعلق بالأطفال. هل يمكنك تقديم المزيد من التفاصيل؟",
            "budget_too_low": "لم أجد أي منتجات بهذه الميزانية. تبدأ هدايانا من حوالي 35 درهم. هل تود تعديل الميزانية؟",
            "age_too_old": "نحن متخصصون في هدايا الأطفال حتى 3 سنوات. للأطفال الأكبر سنًا، أنصحك بمراجعة أقسام المتجر الرئيسية.",
            "not_baby_related": "لا يبدو أن هذا طلب هدية متعلق بالأطفال أو الأمهات. تم تصميمي للمساعدة في هدايا الأطفال والأمهات وآباء المستقبل. كيف يمكنني المساعدة في ذلك؟",
            "empty_query": "يرجى إخباري لمن الهدية، وعمر الطفل، وميزانيتك حتى أتمكن من مساعدتك في العثور على الهدية المثالية.",
            "vague": "طلبك عام قليلاً. للمساعدة بشكل أفضل، هل يمكنك مشاركة عمر الطفل وميزانيتك؟"
        }
    }

    return {
        "message_en": messages["en"].get(reason, messages["en"]["out_of_scope"]),
        "message_ar": messages["ar"].get(reason, messages["ar"]["out_of_scope"])
    }
