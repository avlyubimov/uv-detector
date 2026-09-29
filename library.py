from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from detector import AnalysisError, decode_image, read_color, rgb_to_lab
from models import AnalysisResponse, ColorLibrary, DEFAULT_LIBRARY_ID, Estimate, LibraryId, Match, ReferenceSample, Sampling


LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "libraries.json"


@lru_cache(maxsize=1)
def load_libraries() -> dict[str, ColorLibrary]:
    libraries = TypeAdapter(list[ColorLibrary]).validate_json(LIBRARY_PATH.read_text(encoding="utf-8"))
    return {library.id: library for library in libraries}


def sample_estimate(sample: ReferenceSample, baseline: float) -> Estimate:
    if sample.intensity_uw_cm2 is None:
        transmission = 100.0 - sample.protection_percent
        intensity = baseline * transmission / 100.0
    else:
        intensity = sample.intensity_uw_cm2
        transmission = 100.0 * intensity / baseline
    if not math.isfinite(transmission):
        raise AnalysisError("invalid_reference_intensity", "Контрольная интенсивность слишком мала для расчёта.")
    protection = 100.0 - transmission if transmission <= 100.0 else None
    return Estimate(
        intensity_uw_cm2=round(intensity, 1),
        transmission_percent=round(transmission, 2),
        protection_percent=round(protection, 2) if protection is not None else None,
    )


def analyze_photo(
    content: bytes,
    library_id: LibraryId = DEFAULT_LIBRARY_ID,
    reference_intensity: float | None = None,
    roi: tuple[float, float, float, float] | None = None,
    white_roi: tuple[float, float, float, float] | None = None,
) -> AnalysisResponse:
    library = load_libraries()[library_id]
    baseline = reference_intensity if reference_intensity is not None else library.reference_intensity_uw_cm2
    reading = read_color(decode_image(content), roi, white_roi)
    reference_labs = rgb_to_lab(np.array([sample.rgb for sample in library.samples]))
    distances = np.linalg.norm(reference_labs - reading.lab, axis=1)
    matches = [
        Match(
            sample_id=library.samples[int(index)].id,
            rgb=library.samples[int(index)].rgb,
            delta_e=round(float(distances[index]), 3),
            estimate=sample_estimate(library.samples[int(index)], baseline),
        )
        for index in np.argsort(distances)[:3]
    ]
    messages = [*library.warnings, *reading.warnings]
    status = "estimated"
    estimate = matches[0].estimate
    if reference_intensity is None:
        messages.append(f"Контрольная интенсивность по умолчанию: {baseline:g} мкВт/см². Её можно изменить параметром reference_intensity_uw_cm2.")
    color_outside_library = library.max_color_distance is not None and matches[0].delta_e > library.max_color_distance
    if color_outside_library or reading.color_spread > 12:
        status = "no_match"
        estimate = None
        messages.append("Нет достаточно близкого однородного цвета в библиотеке. Переснимите карту или дополните эталоны.")
    elif matches[0].delta_e > 5 or matches[1].delta_e - matches[0].delta_e < 0.8 or reading.warnings:
        status = "uncertain"
        messages.append("Сопоставление неоднозначно или качество фото недостаточно; результат требует проверки.")
    if estimate is not None and estimate.protection_percent is None:
        status = "uncertain"
        messages.append("Оценка интенсивности выше контрольного облучения. Степень защиты не рассчитана; проверьте контроль и условия опыта.")
    return AnalysisResponse(
        library_id=library_id, status=status, estimate=estimate, measurement_validated=library.calibrated,
        reference_intensity_uw_cm2=baseline,
        reference_intensity_source="request" if reference_intensity is not None else "document_default",
        nearest_matches=matches,
        sampling=Sampling(
            method=reading.method, rgb=reading.rgb, raw_rgb=reading.raw_rgb,
            hex="#" + "".join(f"{channel:02X}" for channel in reading.rgb),
            region_polygon=reading.polygon, white_balance_applied=reading.white_balance_applied,
            color_spread_delta_e=reading.color_spread,
        ),
        warnings=messages,
    )
