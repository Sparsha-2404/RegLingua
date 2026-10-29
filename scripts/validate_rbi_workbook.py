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

def colnum(ref):
    letters = re.match(r"[A-Z]+", ref or "")
    n = 0
    if letters:
        for c in letters.group():
            n = n * 26 + ord(c) - 64
    return n

def main():
    if not FILE.exists():
        print("ERROR: Workbook not found:", FILE)
        return 2
    with zipfile.ZipFile(FILE) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        relroot = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rels = {x.attrib["Id"]: x.attrib["Target"] for x in relroot.findall(P+"Relationship")}
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//"+M+"t")) for si in ss.findall(M+"si")]
        paths = {}
        for s in wb.findall(".//"+M+"sheet"):
            target = rels[s.attrib[R+"id"]]
            paths[s.attrib["name"]] = target.lstrip("/") if target.startswith("/") else (target if target.startswith("xl/") else "xl/"+target)
        def rows(sheet):
            root = ET.fromstring(z.read(paths[sheet]))
            out = []
            for row in root.findall(".//"+M+"sheetData/"+M+"row"):
                vals = {}
                for c in row.findall(M+"c"):
                    idx = colnum(c.attrib.get("r"))
                    v = c.find(M+"v")
                    inline = c.find(M+"is")
                    val = ""
                    if c.attrib.get("t") == "s" and v is not None:
                        val = shared[int(v.text)]
                    elif c.attrib.get("t") == "inlineStr" and inline is not None:
                        val = "".join(t.text or "" for t in inline.findall(".//"+M+"t"))
                    elif v is not None:
                        val = v.text or ""
                    vals[idx] = val
                out.append(vals)
            return out
        raw = rows("RBI")
        regraw = rows("REGULATORY_SOURCES")
    def dictrows(raw):
        if not raw: return []
        headers = {i: v.strip() for i, v in raw[0].items() if v.strip()}
        return [{headers[i]: v.strip() for i, v in row.items() if i in headers} for row in raw[1:] if row]
    ex = [r for r in dictrows(raw) if r.get("ID") and (r.get("TEXT") or r.get("EXAMPLE_TEXT") or r.get("text"))]
    registry = dictrows(regraw)
    def val(row, key):
        return next((v for k, v in row.items() if k.strip().casefold() == key.casefold()), "")
    ids = [val(r, "SOURCE_ID") for r in registry if val(r, "SOURCE_ID")]
    idset = set(ids)
    missing_fields = [(val(r,"ID"), f) for r in ex for f in ("ID","TEXT","RISK LEVEL","LANGUAGE","SOURCE_ID","SECTION","SOURCE_URL") if not val(r,f)]
    missing_ids = [(val(r,"ID"), val(r,"SOURCE_ID")) for r in ex if val(r,"SOURCE_ID") not in idset]
    dupes = sorted({x for x in ids if ids.count(x)>1})
    badurls = [(val(r,"ID"), val(r,"SOURCE_URL")) for r in ex if urlparse(val(r,"SOURCE_URL")).scheme != "https" or not urlparse(val(r,"SOURCE_URL")).netloc]
    print("Workbook:", FILE)
    print("Populated RBI examples:", len(ex))
    print("Missing required fields:", len(missing_fields))
    print("SOURCE_IDs missing from registry:", len(missing_ids))
    print("Duplicate SOURCE_IDs in registry:", len(dupes))
    print("Invalid/non-HTTPS example URLs:", len(badurls))
    for label, items in (("MISSING FIELDS",missing_fields),("MISSING SOURCE IDS",missing_ids),("DUPLICATE REGISTRY IDS",dupes),("INVALID URLS",badurls)):
        print("\n"+label+":")
        print("  None" if not items else "\n".join("  "+str(x) for x in items))
    ok = not (missing_fields or missing_ids or dupes or badurls)
    print("\nSTRUCTURAL RESULT:", "PASS" if ok else "FAIL")
    print("Note: This checks workbook structure and URL format, not URL reachability or legal/evidentiary support.")
    return 0 if ok else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("VALIDATION ERROR:", type(e).__name__, str(e))
        raise SystemExit(2)