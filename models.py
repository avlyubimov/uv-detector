from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


LibraryId = Literal["photo_examples", "document_template"]
DEFAULT_LIBRARY_ID: LibraryId = "document_template"
ColorChannel = Annotated[int, Field(ge=0, le=255)]
RGB = tuple[ColorChannel, ColorChannel, ColorChannel]


class Region(BaseModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def within_image(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Область должна целиком находиться внутри изображения.")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height


class ReferenceSample(BaseModel):
    id: str
    rgb: RGB
    intensity_uw_cm2: float | None = Field(default=None, ge=0)
    protection_percent: float = Field(ge=0, le=100)
    source_protection_percent: float | None = None
    source_transmission_percent: float | None = None
    source_image: str | None = None


class ColorLibrary(BaseModel):
    id: LibraryId
    title: str
    source_document: str
    source_sha256: str
    calibrated: bool = False
    reference_intensity_uw_cm2: float = Field(gt=0)
    white_balance_target: int | None = None
    max_color_distance: float | None = Field(default=12.0, gt=0, allow_inf_nan=False)
    warnings: list[str]
    samples: list[ReferenceSample] = Field(min_length=2)


class Estimate(BaseModel):
    intensity_uw_cm2: float
    transmission_percent: float
    protection_percent: float | None


class Match(BaseModel):
    sample_id: str
    rgb: RGB
    delta_e: float
    estimate: Estimate


class Sampling(BaseModel):
    method: Literal["uv_test_card", "manual_roi"]
    rgb: RGB
    raw_rgb: RGB
    hex: str
    region_polygon: list[list[float]]
    white_balance_applied: bool
    color_spread_delta_e: float


class AnalysisResponse(BaseModel):
    library_id: LibraryId
    status: Literal["estimated", "uncertain", "no_match"]
    measurement_validated: bool = False
    estimate: Estimate | None
    reference_intensity_uw_cm2: float
    reference_intensity_source: Literal["request", "document_default"]
    nearest_matches: list[Match]
    sampling: Sampling
    warnings: list[str]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
