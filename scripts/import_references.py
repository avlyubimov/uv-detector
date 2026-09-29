import argparse
from hashlib import sha256
from pathlib import Path
import posixpath
from xml.etree import ElementTree
from zipfile import ZipFile

from PIL import Image
from pydantic import TypeAdapter

from detector import decode_image, read_color
from models import ColorLibrary, ReferenceSample


NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"


def cell_text(cell: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in cell.findall(".//w:t", NAMESPACES)).strip()


def number(text: str) -> float:
    return float(text.replace(",", ".").replace("%", "").strip())


def import_template(path: Path, baseline: float) -> ColorLibrary:
    samples = []
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        for row in document.findall(".//w:tr", NAMESPACES):
            cells = [cell_text(cell) for cell in row.findall("w:tc", NAMESPACES)]
            if len(cells) < 4 or not cells[2].startswith("#"):
                continue
            protection = number(cells[0])
            red, green, blue = (int(channel.strip()) for channel in cells[3].split(","))
            rgb = (red, green, blue)
            if "#" + "".join(f"{channel:02X}" for channel in rgb) != cells[2].upper():
                raise ValueError(f"HEX и RGB не совпадают для строки {cells[0]}")
            samples.append(ReferenceSample(id=f"protection-{int(protection)}", rgb=rgb, protection_percent=protection))
    if sorted(sample.protection_percent for sample in samples) != list(range(101)):
        raise ValueError("Ожидалась таблица из 101 цвета для защиты 0–100%.")
    return ColorLibrary(
        id="document_template", title="Учебная таблица: 101 цвет",
        source_document=path.name, source_sha256=sha256(path.read_bytes()).hexdigest(),
        reference_intensity_uw_cm2=baseline, max_color_distance=None,
        warnings=[
            "Расчёт по учебной таблице цветов; калиброванные данные будут добавлены после опытов.",
        ], samples=samples,
    )


def import_photos(path: Path, output: Path) -> ColorLibrary:
    samples = []
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        relationships = ElementTree.fromstring(archive.read("word/_rels/document.xml.rels"))
        targets = {
            relation.attrib["Id"]: posixpath.normpath("word/" + relation.attrib["Target"])
            for relation in relationships.findall(f"{{{RELATIONSHIP_NAMESPACE}}}Relationship")
            if relation.attrib.get("TargetMode") != "External"
        }
        for row in document.findall(".//w:tr", NAMESPACES):
            cells = row.findall("w:tc", NAMESPACES)
            if len(cells) < 5 or not cell_text(cells[0]).isdigit():
                continue
            number_id = int(cell_text(cells[0]))
            embedded = cells[1].find(".//a:blip", NAMESPACES)
            if embedded is None:
                raise ValueError(f"В строке {number_id} отсутствует фото.")
            relationship_id = embedded.attrib[f"{{{NAMESPACES['r']}}}embed"]
            pixels = decode_image(archive.read(targets[relationship_id]))
            reading = read_color(pixels)
            filename = f"examples/photo-{number_id}.png"
            image_path = output / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(image_path)
            samples.append(ReferenceSample(
                id=f"photo-{number_id}", rgb=reading.rgb,
                intensity_uw_cm2=number(cell_text(cells[2])),
                protection_percent=number(cell_text(cells[4])),
                source_transmission_percent=number(cell_text(cells[3])),
                source_protection_percent=number(cell_text(cells[4])),
                source_image=filename,
            ))
    controls = [sample for sample in samples if sample.protection_percent == 0]
    if len(samples) != 5 or len(controls) != 1 or not controls[0].intensity_uw_cm2:
        raise ValueError("Ожидались пять фото и один контроль без защиты с ненулевой интенсивностью.")
    baseline = controls[0].intensity_uw_cm2
    for sample in samples:
        sample.protection_percent = round(100.0 * (1.0 - sample.intensity_uw_cm2 / baseline), 2)
    return ColorLibrary(
        id="photo_examples", title="Пять фотоэталонов с подписями из документа",
        source_document=path.name, source_sha256=sha256(path.read_bytes()).hexdigest(),
        reference_intensity_uw_cm2=baseline, white_balance_target=240,
        warnings=[
            "Оценка по пяти фото-примерам; независимая физическая калибровка и точность не подтверждены.",
            "Возвращается ближайший фотоэталон, без интерполяции и без точности 1%. Освещение, время экспозиции и тип карты должны совпадать с калибровкой.",
        ], samples=samples,
    )


def main():
    parser = argparse.ArgumentParser(description="Импорт исходных DOCX в локальную библиотеку УФ-эталонов")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    arguments = parser.parse_args()
    photos = import_photos(arguments.photos, arguments.output)
    template = import_template(arguments.template, photos.reference_intensity_uw_cm2)
    output = TypeAdapter(list[ColorLibrary]).dump_json([photos, template], indent=2)
    (arguments.output / "libraries.json").write_bytes(output + b"\n")
    print(f"Импортировано {len(photos.samples)} фотоэталонов и {len(template.samples)} цветов: {arguments.output}")


if __name__ == "__main__":
    main()
