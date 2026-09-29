from dataclasses import dataclass
from io import BytesIO
from typing import Literal
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_WORKING_SIDE = 1800
CARD_WIDTH = 900
CARD_HEIGHT = 560
TEST_RECT = (0.10, 0.55, 0.80, 0.21)


class AnalysisError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass
class ColorReading:
    rgb: tuple[int, int, int]
    raw_rgb: tuple[int, int, int]
    lab: list[float]
    polygon: list[list[float]]
    method: Literal["uv_test_card", "manual_roi"]
    white_balance_applied: bool
    color_spread: float
    warnings: list[str]


def decode_image(content: bytes) -> np.ndarray:
    if not content:
        raise AnalysisError("empty_file", "Изображение пустое.", 400)
    if len(content) > MAX_FILE_BYTES:
        raise AnalysisError("file_too_large", "Максимальный размер фото — 10 МБ.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise AnalysisError("unsupported_image", "Нужен JPEG, PNG, WebP или BMP.", 415)
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise AnalysisError("image_too_large", "Максимальное разрешение — 24 мегапикселя.", 413)
                if getattr(source, "n_frames", 1) != 1:
                    raise AnalysisError("animated_image", "Отправьте одно неподвижное изображение.", 415)
                oriented = ImageOps.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    oriented = Image.alpha_composite(background, rgba)
                return np.asarray(oriented.convert("RGB")).copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise AnalysisError("image_too_large", "Слишком большое разрешение изображения.", 413) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError("invalid_image", "Не удалось прочитать изображение.", 415) from error


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32).reshape(1, -1, 3) / 255.0
    return cv2.cvtColor(values, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def rgb_tuple(values: np.ndarray) -> tuple[int, int, int]:
    return int(values[0]), int(values[1]), int(values[2])


def crop_rect(image: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    left, top, width, height = rect
    image_height, image_width = image.shape[:2]
    start_x, start_y = round(left * image_width), round(top * image_height)
    end_x, end_y = round((left + width) * image_width), round((top + height) * image_height)
    crop = image[start_y:end_y, start_x:end_x]
    if min(crop.shape[:2]) < 8:
        raise AnalysisError("region_too_small", "Область измерения должна быть не меньше 8×8 пикселей.")
    return crop


def order_corners(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0)
    if np.linalg.norm(ordered[1] - ordered[0]) < np.linalg.norm(ordered[2] - ordered[1]):
        ordered = np.roll(ordered, -1, axis=0)
    return ordered.astype(np.float32)


def scale_position(card: np.ndarray) -> tuple[int, float]:
    pixels = card[:, 70:830].astype(np.float32)
    purple = np.minimum(pixels[:, :, 0], pixels[:, :, 2]) - pixels[:, :, 1]
    profile = np.percentile(purple, 55, axis=1)
    profile = np.convolve(profile, np.ones(9) / 9, mode="same")
    prominence = profile - (np.roll(profile, 40) + np.roll(profile, -40)) / 2
    eligible = np.zeros(CARD_HEIGHT, dtype=bool)
    eligible[130:265] = True
    eligible[295:430] = True
    prominence[~eligible] = -1000
    row = int(np.argmax(prominence))
    return row, float(prominence[row])


def find_card(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original_height, original_width = image.shape[:2]
    resize_factor = min(1.0, MAX_WORKING_SIDE / max(original_height, original_width))
    small = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    chromatic = (hue >= 85) & (hue <= 165) & (saturation >= 45) & (value > 95)
    neutral = (saturation < 45) & (value > 165)
    contours = []
    for mask in (chromatic, neutral, chromatic | neutral):
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(sorted(found, key=cv2.contourArea, reverse=True)[:5])
    candidates = []
    destination = np.float32([[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < small.shape[0] * small.shape[1] * 0.015:
            continue
        rectangle = cv2.minAreaRect(contour)
        side_lengths = rectangle[1]
        if min(side_lengths) < 90 or not 1.3 <= max(side_lengths) / min(side_lengths) <= 2.1:
            continue
        if area / (side_lengths[0] * side_lengths[1]) < 0.72:
            continue
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, perimeter * 0.025, True)
        if len(corners) != 4:
            corners = cv2.boxPoints(rectangle)
        corners = order_corners(corners.reshape(4, 2)) / resize_factor
        transform = cv2.getPerspectiveTransform(corners, destination)
        card = cv2.warpPerspective(image, transform, (CARD_WIDTH, CARD_HEIGHT))
        row, prominence = scale_position(card)
        if prominence < 8:
            continue
        if row > CARD_HEIGHT // 2:
            corners = np.roll(corners, -2, axis=0)
            transform = cv2.getPerspectiveTransform(corners, destination)
            card = cv2.warpPerspective(image, transform, (CARD_WIDTH, CARD_HEIGHT))
        center = corners.mean(axis=0)
        if any(np.linalg.norm(center - candidate[3]) < min(side_lengths) * 0.2 / resize_factor for candidate in candidates):
            continue
        candidates.append((prominence, card, transform, center))
    if not candidates:
        raise AnalysisError("card_not_found", "Не найдена UV-TEST CARD. Снимите всю карту крупно или передайте roi области TEST AREA.")
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * 0.8:
        raise AnalysisError("multiple_cards", "В кадре несколько похожих карт. Оставьте одну или задайте roi.")
    _, card, transform, _ = candidates[0]
    return card, transform


def estimate_white(card: np.ndarray) -> np.ndarray:
    border = np.concatenate([
        card[15:45, 30:-30].reshape(-1, 3),
        card[30:-30, 15:55].reshape(-1, 3),
        card[30:-30, -55:-15].reshape(-1, 3),
    ])
    brightness = border.mean(axis=1)
    return np.median(border[brightness >= np.percentile(brightness, 50)], axis=0)


def read_color(
    image: np.ndarray,
    roi: tuple[float, float, float, float] | None = None,
    white_roi: tuple[float, float, float, float] | None = None,
) -> ColorReading:
    messages = []
    if roi is None:
        card, transform = find_card(image)
        sample = crop_rect(card, TEST_RECT)
        white = estimate_white(card)
        left, top, width, height = TEST_RECT
        polygon = np.float32([[
            [left * CARD_WIDTH, top * CARD_HEIGHT],
            [(left + width) * CARD_WIDTH, top * CARD_HEIGHT],
            [(left + width) * CARD_WIDTH, (top + height) * CARD_HEIGHT],
            [left * CARD_WIDTH, (top + height) * CARD_HEIGHT],
        ]])
        polygon = cv2.perspectiveTransform(polygon, np.linalg.inv(transform))[0]
        polygon /= np.float32([image.shape[1], image.shape[0]])
        method = "uv_test_card"
    else:
        sample = crop_rect(image, roi)
        white = None
        left, top, width, height = roi
        polygon = np.array([[left, top], [left + width, top], [left + width, top + height], [left, top + height]])
        method = "manual_roi"
    if white_roi is not None:
        white = np.median(crop_rect(image, white_roi).reshape(-1, 3), axis=0)
    raw_pixels = sample.reshape(-1, 3)
    raw_rgb = np.median(raw_pixels, axis=0)
    pixels = raw_pixels.astype(np.float32)
    if white is not None:
        if np.min(white) < 65:
            raise AnalysisError("invalid_white_reference", "Белая область слишком тёмная. Переснимите при более равномерном освещении.")
        if np.max(white) >= 254:
            messages.append("Белая область частично пересвечена; коррекция цвета приблизительная.")
        pixels = np.clip(pixels * (240.0 / white), 0, 255)
    else:
        messages.append("Баланс белого не скорректирован: передайте white_roi белой области карты.")
    median_rgb = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels - median_rgb, axis=1)
    pixels = pixels[distances <= np.percentile(distances, 80)]
    rgb = np.rint(np.median(pixels, axis=0)).astype(int)
    labs = rgb_to_lab(pixels[::max(1, len(pixels) // 4000)])
    lab = rgb_to_lab(rgb)[0]
    spread = float(np.median(np.linalg.norm(labs - lab, axis=1)))
    if spread > 5:
        messages.append("Цвет неоднороден: возможны тени, блики или неверно выделенная область.")
    if float(np.mean(np.any(raw_pixels >= 253, axis=1))) > 0.15:
        messages.append("На тестовой области есть пересвет; рекомендуется новое фото.")
    if min(image.shape[:2]) < 400:
        messages.append("Низкое разрешение фото может снижать точность выделения карты.")
    return ColorReading(
        rgb=rgb_tuple(rgb), raw_rgb=rgb_tuple(np.rint(raw_rgb)),
        lab=lab.tolist(), polygon=np.round(polygon, 5).tolist(), method=method,
        white_balance_applied=white is not None, color_spread=round(spread, 2), warnings=messages,
    )
