"""
Pydantic schemas for structured output validation.
All outputs must validate against these schemas.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional


class GiftRecommendation(BaseModel):
    """Single gift recommendation with bilingual support."""
    product_id: str = Field(..., description="Product ID from catalog")
    product_name_en: str = Field(..., description="Product name in English")
    product_name_ar: str = Field(..., description="Product name in Arabic")
    price_aed: float = Field(..., ge=0, description="Price in UAE Dirhams")
    category_en: str = Field(..., description="Category in English")
    category_ar: str = Field(..., description="Category in Arabic")
    reason_en: str = Field(..., min_length=10, description="Why this fits the query in English")
    reason_ar: str = Field(..., min_length=10, description="Why this fits the query in Arabic")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    age_suitability: str = Field(..., description="Age range this product suits")
    in_stock: bool = Field(default=True, description="Whether product is in stock")

    @field_validator('reason_en')
    @classmethod
    def reason_not_generic_en(cls, v):
        generic = ["good product", "nice gift", "great choice", "perfect gift"]
        if any(g in v.lower() for g in generic) and len(v) < 30:
            raise ValueError("Reason too generic, must be specific")
        return v

    @field_validator('reason_ar')
    @classmethod
    def reason_not_generic_ar(cls, v):
        # Brand names can be Latin, but Arabic reasons should not be mostly transliterated.
        latin_chars = sum(1 for char in v.lower() if char in "abcdefghijklmnopqrstuvwxyz")
        if latin_chars > max(8, len(v) * 0.2):
            raise ValueError("Arabic reason contains too much Latin text - likely transliterated")
        return v


class GiftFinderResponse(BaseModel):
    """Top-level response schema."""
    query_understood_en: str = Field(..., description="LLM interpretation of request in English")
    query_understood_ar: str = Field(..., description="LLM interpretation of request in Arabic")
    recommendations: List[GiftRecommendation] = Field(default_factory=list)
    out_of_scope: bool = Field(default=False, description="True if query cannot be fulfilled")
    uncertainty_note_en: Optional[str] = Field(None, description="Explanation when confidence is low")
    uncertainty_note_ar: Optional[str] = Field(None, description="Explanation in Arabic when confidence is low")
    language_detected: str = Field(..., pattern="^(en|ar)$")
    budget_extracted: Optional[float] = Field(None, description="Budget extracted from query")
    age_months_extracted: Optional[int] = Field(None, description="Age in months extracted from query")

    @model_validator(mode="after")
    def validate_uncertainty_handling(self):
        """Make uncertainty explicit for refusals and weak matches."""
        if self.out_of_scope:
            if self.recommendations:
                raise ValueError("out_of_scope=True but recommendations not empty")
            if not self.uncertainty_note_en or not self.uncertainty_note_ar:
                raise ValueError("Out-of-scope responses must include bilingual uncertainty notes")
            return self

        if not self.recommendations:
            raise ValueError("out_of_scope=False but no recommendations provided")

        if self.recommendations:
            min_confidence = min(rec.confidence for rec in self.recommendations)
            if min_confidence < 0.7 and not self.uncertainty_note_en:
                raise ValueError("Low-confidence recommendations must include an uncertainty note")

        return self
