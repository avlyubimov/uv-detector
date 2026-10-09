import logging
import math
import os
from io import BytesIO
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image
from pydantic import ValidationError
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from detector import MAX_FILE_BYTES
from models import AnalysisResponse, DEFAULT_LIBRARY_ID, RGB


logger = logging.getLogger(__name__)


async def open_api_client(application: Application, transport: httpx.AsyncBaseTransport | None = None):
    headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
    application.bot_data["api_client"] = httpx.AsyncClient(
        base_url=os.getenv("UV_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        headers=headers, timeout=45.0, transport=transport,
    )


async def close_api_client(application: Application):
    await application.bot_data["api_client"].aclose()


async def start(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message:
        await update.effective_message.reply_text(
            "Отправьте фото одной UV-TEST CARD целиком, крупно и без бликов. "
            "Лучше отправлять изображение как файл без сжатия.\n\n"
            "Я найду цвет TEST AREA в таблице цветов и покажу интенсивность, "
            "пропускание и защиту по учебной шкале.\n\n"
            "Контрольная интенсивность: 1500 мкВт/см². Изменить: /reference 1500."
        )


async def set_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return
    try:
        if len(context.args) != 1:
            raise ValueError
        intensity = float(context.args[0].replace(",", "."))
        if not math.isfinite(intensity) or not 0 < intensity <= 1_000_000:
            raise ValueError
    except ValueError:
        await message.reply_text("Укажите положительную интенсивность без барьера: /reference 1500 (мкВт/см²).")
        return
    context.chat_data["reference_intensity_uw_cm2"] = intensity
    await message.reply_text(f"Контроль для этого чата: {intensity:g} мкВт/см². Настройка действует до перезапуска бота.")


def format_result(result: AnalysisResponse) -> str:
    if result.estimate is None:
        return "Не удалось определить цвет тестовой области. Переснимите одну карту крупно, при ровном освещении и без бликов."
    estimate = result.estimate
    protection = f"≈ {estimate.protection_percent:g}%" if estimate.protection_percent is not None else "не определена"
    lines = [
        "Результат по таблице цветов:" if result.library_id == "document_template" else "Результат по фотоэталонам:",
        f"Интенсивность: ≈ {estimate.intensity_uw_cm2:g} мкВт/см²",
        f"Пропускание: ≈ {estimate.transmission_percent:g}%",
        f"Защита: {protection}",
        "Цвет полоски — на образце выше.",
    ]
    if result.status == "uncertain":
        lines.append("Выбран ближайший оттенок; совпадение приблизительное.")
    if estimate.protection_percent is None:
        lines.append("Проверьте контрольную интенсивность: она ниже оценки для полоски.")
    if not result.measurement_validated:
        lines.extend(["", "Расчёт по учебной шкале."])
    return "\n".join(lines)


def make_color_sample(rgb: RGB) -> bytes:
    output = BytesIO()
    with Image.new("RGB", (256, 256), (160, 160, 160)) as sample:
        sample.paste(rgb, (8, 8, 248, 248))
        sample.save(output, format="PNG")
    return output.getvalue()


async def analyze_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return
    attachment = message.photo[-1] if message.photo else message.document
    if attachment is None:
        return
    if attachment.file_size and attachment.file_size > MAX_FILE_BYTES:
        await message.reply_text("Максимальный размер фото — 10 МБ.")
        return
    try:
        telegram_file = await attachment.get_file()
        content = bytes(await telegram_file.download_as_bytearray())
        if len(content) > MAX_FILE_BYTES:
            await message.reply_text("Максимальный размер фото — 10 МБ.")
            return
        data = {"library_id": DEFAULT_LIBRARY_ID}
        if "reference_intensity_uw_cm2" in context.chat_data:
            data["reference_intensity_uw_cm2"] = str(context.chat_data["reference_intensity_uw_cm2"])
        client = context.bot_data["api_client"]
        response = await client.post("/analyze", files={"file": ("photo", content, "application/octet-stream")}, data=data)
        if response.status_code in {400, 413, 415, 422}:
            detail = response.json().get("detail", {})
            explanation = detail.get("message", "Проверьте изображение и параметры.") if isinstance(detail, dict) else "Проверьте изображение и параметры."
            await message.reply_text(explanation)
            return
        response.raise_for_status()
        result = AnalysisResponse.model_validate(response.json())
    except (httpx.HTTPError, TelegramError, ValidationError, ValueError):
        logger.warning("Не удалось скачать фото или получить ответ API.")
        await message.reply_text("Сервис анализа временно недоступен. Попробуйте позже.")
        return
    if result.estimate is None:
        await message.reply_text(format_result(result)[:4000])
    else:
        await message.reply_photo(
            photo=make_color_sample(result.sampling.rgb),
            caption=format_result(result)[:1024],
            filename="uv-color.png",
        )


async def on_error(_update: object, _context: ContextTypes.DEFAULT_TYPE):
    logger.error("Ошибка обработки Telegram update; проверьте доступность Telegram и API.")


def build_application(token: str, *, webhook: bool = False) -> Application:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    builder = (
        Application.builder().token(token)
        .post_init(open_api_client).post_shutdown(close_api_client)
    )
    if webhook:
        builder = builder.updater(None)
    application = builder.build()
    application.add_handler(CommandHandler(["start", "help"], start))
    application.add_handler(CommandHandler("reference", set_reference))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, analyze_message))
    application.add_error_handler(on_error)
    return application


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Добавьте TELEGRAM_BOT_TOKEN от @BotFather в .env или переменные окружения.")
    logging.basicConfig(level=logging.WARNING)
    application = build_application(token)
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
