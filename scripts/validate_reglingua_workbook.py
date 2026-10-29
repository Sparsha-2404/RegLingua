
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "evaluation" / "RegLingua_M1_FINAL.xlsx"

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/package/2006/relationships}"

REQUIRED = [
    "ID", "TEXT", "LANGUAGE", "REGULATOR", "TOPIC", "RISK LEVEL",
    "SOURCE", "SOURCE REFERENCE", "SOURCE_ID", "DOCUMENT_TITLE",
    "SECTION", "EFFECTIVE_DATE", "SOURCE_URL",
]


def colnum(ref):
    match = re.match(r"[A-Z]+", ref or "")
    result = 0
    if match:
        for char in match.group():
            result = result * 26 + ord(char) - 64
    return result


def main():
    if not FILE.exists():
        print("ERROR: Workbook not found:", FILE)
        return 2

    with zipfile.ZipFile(FILE) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        rels = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(P + "Relationship")
        }

        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in item.findall(".//" + M + "t"))
                for item in root.findall(M + "si")
            ]

        sheet_paths = {}
        for sheet in workbook.findall(".//" + M + "sheet"):
            target = rels[sheet.attrib[R + "id"]]
            if target.startswith("/"):
                path = target.lstrip("/")
            elif target.startswith("xl/"):
                path = target
            else:
                path = "xl/" + target
            sheet_paths[sheet.attrib["name"]] = path

        def read_rows(sheet_name):
            root = ET.fromstring(archive.read(sheet_paths[sheet_name]))
            result = []
            for row in root.findall(".//" + M + "sheetData/" + M + "row"):
                values = {}
                for cell in row.findall(M + "c"):
                    column = colnum(cell.attrib.get("r", ""))
                    if not column:
                        continue
                    kind = cell.attrib.get("t")
                    value = cell.find(M + "v")
                    if kind == "s" and value is not None:
                        text = shared[int(value.text)]
                    elif kind == "inlineStr":
                        inline = cell.find(M + "is")
                        text = (
                            "".join(t.text or "" for t in inline.findall(".//" + M + "t"))
                            if inline is not None else ""
                        )
                    else:
                        text = (value.text or "") if value is not None else ""
                    values[column] = text
                result.append(values)
            return result

        def to_dicts(raw):
            if not raw:
                return []
            headers = {
                index: value.strip()
                for index, value in raw[0].items()
                if value.strip()
            }
            result = []
            for row in raw[1:]:
                mapped = {
                    headers[index]: value.strip()
                    for index, value in row.items()
                    if index in headers
                }
                if any(mapped.values()):
                    result.append(mapped)
            return result

        def val(row, key):
            return next(
                (value for name, value in row.items()
                 if name.strip().casefold() == key.casefold()),
                "",
            )

        if "REGULATORY_SOURCES" not in sheet_paths:
            print("ERROR: REGULATORY_SOURCES sheet not found.")
            return 2

        registry = to_dicts(read_rows("REGULATORY_SOURCES"))
        registry_ids = [
            val(row, "SOURCE_ID")
            for row in registry
            if val(row, "SOURCE_ID")
        ]
        registry_id_set = set(registry_ids)
        duplicates = sorted({
            source_id for source_id in registry_ids
            if registry_ids.count(source_id) > 1
        })

        overall_ok = True
        print("Workbook:", FILE)

        for sheet_name in ("SEBI", "RBI", "DPDP"):
            if sheet_name not in sheet_paths:
                print(f"\n{sheet_name}: worksheet not found")
                overall_ok = False
                continue

            examples = [
                row for row in to_dicts(read_rows(sheet_name))
                if val(row, "ID")
                and (val(row, "TEXT") or val(row, "EXAMPLE_TEXT"))
            ]

            missing_fields = [
                (val(row, "ID"), field)
                for row in examples
                for field in REQUIRED
                if not val(row, field)
            ]
            missing_source_ids = [
                (val(row, "ID"), val(row, "SOURCE_ID"))
                for row in examples
                if not val(row, "SOURCE_ID")
                or val(row, "SOURCE_ID") not in registry_id_set
            ]
            invalid_urls = []
            for row in examples:
                url = val(row, "SOURCE_URL")
                parsed = urlparse(url)
                if parsed.scheme != "https" or not parsed.netloc:
                    invalid_urls.append((val(row, "ID"), url))

            passed = not (
                missing_fields or missing_source_ids
                or duplicates or invalid_urls
            )
            overall_ok = overall_ok and passed

            print(f"\n{sheet_name}: {len(examples)} populated examples")
            print("Missing required fields:", len(missing_fields))
            print("SOURCE_IDs missing from registry:", len(missing_source_ids))
            print("Duplicate SOURCE_IDs in registry:", len(duplicates))
            print("Invalid/non-HTTPS URLs:", len(invalid_urls))

            for label, items in (
                ("MISSING FIELDS", missing_fields),
                ("MISSING SOURCE IDS", missing_source_ids),
                ("INVALID URLS", invalid_urls),
            ):
                if items:
                    print(label, "(first 10):", items[:10])

            print("RESULT:", "PASS" if passed else "FAIL")

        print("\nDuplicate registry IDs:", duplicates[:20] if duplicates else "None")
        print("OVERALL STRUCTURAL RESULT:", "PASS" if overall_ok else "FAIL")
        print(
            "Note: Structural checks do not establish URL reachability "
            "or legal/evidentiary support."
        )
        return 0 if overall_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("VALIDATION ERROR:", type(error).__name__, str(error))
        raise SystemExit(2)
