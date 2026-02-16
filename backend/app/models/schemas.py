from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Set
from enum import Enum

# Configure strict validation globally
class StrictBaseModel(BaseModel):
    """Base model with strict validation - rejects extra fields"""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

# Enums for categorical validation
class BodyType(str, Enum):
    SLIM = "slim"
    ATHLETIC = "athletic"
    AVERAGE = "average"
    CURVY = "curvy"
    PLUS_SIZE = "plus_size"

class Ethnicity(str, Enum):
    ASIAN = "asian"
    BLACK = "black"
    HISPANIC = "hispanic"
    WHITE = "white"
    MIDDLE_EASTERN = "middle_eastern"
    MIXED = "mixed"

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class ClothingCategory(str, Enum):
    """Garment types matching Leffa's supported categories."""
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    DRESSES = "dresses"

# User schemas
class UserCreate(StrictBaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=13, le=120)
    height_cm: float = Field(..., ge=140, le=220)
    weight_kg: float = Field(..., ge=40, le=200)
    gender: Optional[Gender] = None
    style_preference: Optional[str] = Field(None, max_length=500)
    
    @field_validator('height_cm')
    @classmethod
    def validate_height(cls, v: float) -> float:
        if not 140 <= v <= 220:
            raise ValueError("Height must be between 140-220 cm")
        return v
    
    @field_validator('weight_kg')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 40 <= v <= 200:
            raise ValueError("Weight must be between 40-200 kg")
        return v

class UserResponse(UserCreate):
    id: str
    is_full_body: bool = True

# Try-on schemas
class TryOnRequest(StrictBaseModel):
    user_id: str = Field(..., min_length=1)
    clothing_image_url: str = Field(..., min_length=1, max_length=2048)
    clothing_category: ClothingCategory = ClothingCategory.UPPER_BODY
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TryOnResponse(StrictBaseModel):
    result_url: str
    processing_time: float = Field(..., ge=0)
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Body analysis schemas
class BodyAnalysisResponse(StrictBaseModel):
    body_type: str = Field(..., pattern="^(full_body|partial_body)$")
    is_full_body: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    coverage_metric: float = Field(..., ge=0.0)
    error: Optional[str] = None

# Body generation schemas
class BodyGenerationRequest(StrictBaseModel):
    ethnicity: Ethnicity
    body_type: BodyType
    height_cm: float = Field(..., ge=140, le=220)
    weight_kg: float = Field(..., ge=40, le=200)
    skin_tone: str = Field(..., min_length=1, max_length=50)
    count: int = Field(default=4, ge=1, le=10)
    
    @field_validator('height_cm')
    @classmethod
    def validate_height(cls, v: float) -> float:
        if not 140 <= v <= 220:
            raise ValueError("Height must be between 140-220 cm")
        return v
    
    @field_validator('weight_kg')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 40 <= v <= 200:
            raise ValueError("Weight must be between 40-200 kg")
        return v

class BodyGenerationResponse(StrictBaseModel):
    result_urls: List[str]
    processing_time: float = Field(..., ge=0)
    count: int = Field(..., ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Full body generation schemas (from partial image)
class FullBodyGenerationRequest(StrictBaseModel):
    ethnicity: Optional[Ethnicity] = None  # Optional, Gemini can infer
    body_type: BodyType
    height_cm: Optional[float] = Field(None, ge=140, le=220)
    weight_kg: Optional[float] = Field(None, ge=40, le=200)
    gender: Optional[Gender] = None
    pose: str = Field(default="standing", max_length=100)
    clothing: str = Field(default="casual minimal clothing", max_length=200)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: int = Field(default=42, ge=0)
    
    @field_validator('height_cm')
    @classmethod
    def validate_height(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 140 <= v <= 220:
            raise ValueError("Height must be between 140-220 cm")
        return v
    
    @field_validator('weight_kg')
    @classmethod
    def validate_weight(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 40 <= v <= 200:
            raise ValueError("Weight must be between 40-200 kg")
        return v

class FullBodyGenerationResponse(StrictBaseModel):
    result_url: str
    request_id: str
    processing_time: float = Field(..., ge=0)
    face_analysis: Dict[str, Any] = Field(default_factory=dict)
    generation_prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Recommendation schemas
class RecommendationRequest(StrictBaseModel):
    user_photo: bytes
    wardrobe_images: Optional[List[bytes]] = None
    generated_images: Optional[List[bytes]] = None
    max_results: int = Field(default=10, ge=1, le=50)
    
    @field_validator('user_photo')
    @classmethod
    def validate_user_photo_size(cls, v: bytes) -> bytes:
        if len(v) > 10 * 1024 * 1024:  # 10MB
            raise ValueError("User photo size exceeds 10MB limit")
        return v
    
    @field_validator('wardrobe_images')
    @classmethod
    def validate_wardrobe_images_size(cls, v: Optional[List[bytes]]) -> Optional[List[bytes]]:
        if v:
            for img in v:
                if len(img) > 10 * 1024 * 1024:  # 10MB
                    raise ValueError("Wardrobe image size exceeds 10MB limit")
        return v
    
    @field_validator('generated_images')
    @classmethod
    def validate_generated_images_size(cls, v: Optional[List[bytes]]) -> Optional[List[bytes]]:
        if v:
            for img in v:
                if len(img) > 10 * 1024 * 1024:  # 10MB
                    raise ValueError("Generated image size exceeds 10MB limit")
        return v

class ProductRecommendation(StrictBaseModel):
    id: str
    name: str = Field(..., min_length=1, max_length=500)
    image_url: str = Field(..., min_length=1, max_length=2048)
    price: float = Field(..., ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    category: str = Field(..., min_length=1, max_length=100)
    ebay_url: str = Field(..., min_length=1, max_length=2048)
    relevance_score: float = Field(..., ge=0, le=1)

class RecommendationResponse(StrictBaseModel):
    recommendations: List[ProductRecommendation]
    processing_time: float = Field(..., ge=0)
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Health monitoring schemas
class ComponentHealth(StrictBaseModel):
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class HealthStatus(StrictBaseModel):
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    checks: Dict[str, ComponentHealth]
    timestamp: str

# Error response schema
class ErrorResponse(StrictBaseModel):
    error_code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    details: Optional[Dict[str, Any]] = None
    request_id: str = Field(..., min_length=1)

# Garment management schemas
class GarmentMetadata(StrictBaseModel):
    id: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1, max_length=2048)
    name: str = Field(..., min_length=1, max_length=255)
    path: str = Field(..., min_length=1, max_length=500)
    uploaded_at: str = Field(..., min_length=1)
    size_mb: Optional[float] = Field(None, ge=0)
    size_bytes: Optional[int] = Field(None, ge=0)
    content_type: Optional[str] = Field(None, max_length=100)

class GarmentUploadResponse(StrictBaseModel):
    message: str = Field(..., min_length=1)
    garment: GarmentMetadata

class GarmentListResponse(StrictBaseModel):
    garments: List[GarmentMetadata]
    count: int = Field(..., ge=0)
    user_id: str = Field(..., min_length=1)

class GarmentDeleteResponse(StrictBaseModel):
    message: str = Field(..., min_length=1)
    garment_id: str = Field(..., min_length=1)
    deleted: bool

# Configuration schemas
class ModelConfig(StrictBaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., min_length=1, max_length=50)
    path: str = Field(..., min_length=1, max_length=500)
    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")
    quantization: Optional[str] = Field(None, pattern="^(int8|fp16)$")
    batch_size: int = Field(default=1, ge=1, le=32)
    max_batch_wait_ms: int = Field(default=2000, ge=0, le=10000)

class RateLimitConfig(StrictBaseModel):
    endpoint_pattern: str = Field(..., min_length=1)
    requests_per_minute: int = Field(..., ge=1, le=10000)
    burst_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    per_user: bool = True
    per_ip: bool = True

class SecurityConfig(StrictBaseModel):
    enforce_https: bool = True
    allowed_origins: List[str] = Field(..., min_length=1)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_mime_types: Set[str] = Field(
        default={"image/jpeg", "image/png", "image/webp"}
    )
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440)
    csrf_enabled: bool = True
    
    @field_validator('allowed_origins')
    @classmethod
    def validate_allowed_origins(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one allowed origin must be specified")
        return v

class AppConfig(StrictBaseModel):
    environment: str = Field(..., pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    security: SecurityConfig
    rate_limits: List[RateLimitConfig]
    models: List[ModelConfig]
    
    @field_validator('security')
    @classmethod
    def validate_security_production(cls, v: SecurityConfig, info) -> SecurityConfig:
        # Access environment from validation context
        environment = info.data.get('environment')
        if environment == 'production':
            if not v.enforce_https:
                raise ValueError("HTTPS must be enforced in production")
            if "*" in v.allowed_origins:
                raise ValueError("Wildcard CORS not allowed in production")
        return v

# 3D Generation schemas
class OutputFormat(str, Enum):
    GLB = "glb"
    OBJ = "obj"
    PLY = "ply"

class ThreeDGenerationRequest(StrictBaseModel):
    output_format: OutputFormat = OutputFormat.GLB
    use_segmentation: bool = True
    mc_resolution: int = Field(default=256, ge=128, le=512)
    
    @field_validator('mc_resolution')
    @classmethod
    def validate_mc_resolution(cls, v: int) -> int:
        # Warn if resolution is too high for 4GB VRAM
        if v > 256:
            # This is just validation, actual warning should be in the endpoint
            pass
        return v

class ThreeDGenerationResponse(StrictBaseModel):
    download_token: str = Field(..., min_length=1)
    download_url: str = Field(..., min_length=1)
    expires_at: str = Field(..., min_length=1)
    format: str = Field(..., pattern="^(glb|obj|ply)$")
    file_size_bytes: int = Field(..., ge=0)
    processing_time: float = Field(..., ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
