import os, re, io, sys, gzip, csv, argparse, zipfile
from datetime import datetime
from typing import List, Dict, Iterable
from lxml import etree

AGENCY_KEYWORDS = [
    "FEDERAL RESERVE SYSTEM",
    "DEPARTMENT OF THE TREASURY",
]

def collapse(s: str) -> str:
    import re
    if not s: return ""
    return re.sub(r"\s+", " ", s).strip()

def first_text(tree: etree._ElementTree, xp: str) -> str:
    res = tree.xpath(xp)
    if not res:
        return ""
    node = res[0]
    if isinstance(node, etree._Element):
        return collapse(" ".join(node.itertext()))
    return collapse(str(node))

def all_texts(tree: etree._ElementTree, xp: str) -> List[str]:
    out = []
    for node in tree.xpath(xp):
        if isinstance(node, etree._Element):
            out.append(collapse(" ".join(node.itertext())))
        else:
            out.append(collapse(str(node)))
    return [t for t in out if t]

def parse_one_xml(xml_bytes: bytes, fallback_date: str = None) -> Dict[str, str]:
    tree = etree.parse(io.BytesIO(xml_bytes))

    # publication date
    pub = first_text(tree, "//DATE")
    if not pub and fallback_date:
        pub = fallback_date

    pub_iso = ""
    if pub:
        try:
            pub_iso = datetime.strptime(pub, "%A, %B %d, %Y").strftime("%Y-%m-%d")
        except Exception:
            m = re.search(r"\d{4}-\d{2}-\d{2}", pub)
            pub_iso = m.group(0) if m else ""

    # agencies (upper-cased, deduped)
    agencies = [a.upper() for a in (all_texts(tree, "//AGENCY") + all_texts(tree, "//SUBAGY"))]
    seen = set(); agy_list = []
    for a in agencies:
        if a and a not in seen:
            seen.add(a); agy_list.append(a)

    # identifiers
    frdoc = first_text(tree, "//FRDOC")
    gid = first_text(tree, "//GID")
    doc_id = frdoc or gid

    # title & type
    title = first_text(tree, "//DOCTITLE") or first_text(tree, "//SUBJECT")
    action = first_text(tree, "//ACT") or first_text(tree, "//ACTION")

    # body text: summary + supplementary, else preamble
    summary = collapse(" ".join(all_texts(tree, "//SUM")))
    suppl   = collapse(" ".join(all_texts(tree, "//SUPLINF")))
    preamb  = collapse(" ".join(all_texts(tree, "//PREAMB")))
    text    = " ".join([t for t in [summary, suppl] if t]) if (summary or suppl) else preamb
    if not text:
        text = collapse(" ".join(all_texts(tree, "//PREAMB//P")))

    return {
        "id": doc_id,
        "publication_date": pub_iso or pub,
        "agencies": "; ".join(agy_list),
        "title": title,
        "type": collapse(action),
        "text": text,
    }

def matches_target_agency(agencies_field: str, keywords: Iterable[str]) -> bool:
    if not agencies_field:
        return False
    u = agencies_field.upper()
    return any(k in u for k in keywords)

def process_zip(zip_path: str, out_dir: str, agency_keywords: List[str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(zip_path))[0] + "_fed_treasury.csv.gz"
    out_path = os.path.join(out_dir, out_name)

    with zipfile.ZipFile(zip_path, "r") as zf, gzip.open(out_path, "wt", newline="", encoding="utf-8") as gz:
        writer = csv.DictWriter(gz, fieldnames=["id","publication_date","agencies","title","type","text","source_file"])
        writer.writeheader()

        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        for xp in xml_names:
            data = zf.read(xp)
            # fallback date from filename, e.g., FR-2025-01-02.xml
            m = re.search(r"FR-(\d{4}-\d{2}-\d{2})\.xml", xp)
            fb_date = m.group(1) if m else None
            rec = parse_one_xml(data, fallback_date=fb_date)
            if matches_target_agency(rec.get("agencies",""), agency_keywords):
                rec["source_file"] = xp
                writer.writerow(rec)

    return out_path

def main():
    ap = argparse.ArgumentParser(description="Parse GovInfo Federal Register ZIP(s) to CSV (Fed + Treasury).")
    ap.add_argument("inputs", nargs="+", help="Paths to FR-YYYY.zip or FR-YYYY-MM.zip files")
    ap.add_argument("--out-dir", default="data/processed", help="Output directory for CSVs")
    ap.add_argument("--all-agencies", action="store_true", help="Do not filter; output all agencies")
    args = ap.parse_args()

    kws = [] if args.all_agencies else AGENCY_KEYWORDS
    for zp in args.inputs:
        out = process_zip(zp, args.out_dir, kws if kws else [""])  # pass dummy keyword if unfiltered
        print("Wrote:", out)

if __name__ == "__main__":
    main()
