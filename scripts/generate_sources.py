"""Regenerate data/sources.yaml from the lists below.

The registry is data: review the generated YAML, don't hand-edit it. Every URL here was
checked to resolve (HTTP 200) when added. eCFR requests pin a point-in-time date per tax
year so re-ingestion is reproducible; bump the 2026 date to pick up later amendments.
"""

from pathlib import Path
from urllib.parse import quote

import yaml

OUT = Path(__file__).resolve().parents[1] / "data" / "sources.yaml"
ECFR_DATES = {2025: "2025-12-31", 2026: "2026-09-08"}
PASS_THROUGH = ["sole_prop", "partnership", "s_corp"]

IRC = {
    "132": [], "162": [], "167": [], "168": [], "174": [], "174A": [], "179": [], "183": [],
    "195": [], "197": [], "199A": PASS_THROUGH, "262": [], "263": [], "263A": [], "274": [],
    "280A": PASS_THROUGH, "280F": [], "404": [],
}
IRC_OVERRIDES = {"162": {"IRC § 162(l)": PASS_THROUGH}}  # self-employed health insurance

REGS = [
    "1.132-1", "1.132-5", "1.132-6", "1.132-7",
    "1.162-1", "1.162-2", "1.162-3", "1.162-4", "1.162-5", "1.162-7", "1.162-9", "1.162-10",
    "1.162-11", "1.162-15", "1.162-20", "1.162-21",
    "1.167(a)-1", "1.167(a)-3", "1.168(k)-1", "1.168(k)-2",
    "1.179-1", "1.179-2", "1.179-3", "1.179-4", "1.179-5",
    "1.183-1", "1.183-2", "1.195-1", "1.197-2",
    "1.199A-1", "1.199A-2", "1.199A-3", "1.199A-4", "1.199A-5", "1.199A-6",
    "1.263(a)-1", "1.263(a)-2", "1.263(a)-3", "1.263(a)-4", "1.263(a)-5", "1.263(a)-6",
    "1.263A-1",
    "1.274-1", "1.274-2", "1.274-5T", "1.274-6", "1.274-9", "1.274-10", "1.274-11",
    "1.274-12", "1.274-13", "1.274-14",
    "1.280F-2T", "1.280F-6",
]

IRS = "https://www.irs.gov"
# (id, title, citation root, doc type, entity types, {year: (url, parser)})
PUBLICATIONS = [
    ("pub-334", "Publication 334, Tax Guide for Small Business", "Pub 334", "irs_publication",
     ["sole_prop"], {2025: (f"{IRS}/publications/p334", "irs_html")}),
    ("pub-463", "Publication 463, Travel, Gift, and Car Expenses", "Pub 463", "irs_publication",
     [], {2025: (f"{IRS}/publications/p463", "irs_html")}),
    ("pub-946", "Publication 946, How To Depreciate Property", "Pub 946", "irs_publication",
     [], {2025: (f"{IRS}/publications/p946", "irs_html")}),
    ("pub-587", "Publication 587, Business Use of Your Home", "Pub 587", "irs_publication",
     ["sole_prop", "partnership"], {2025: (f"{IRS}/publications/p587", "irs_html")}),
    # The live page is already the 2026 edition; 2025 survives only as a PDF.
    ("pub-15-b", "Publication 15-B, Employer's Tax Guide to Fringe Benefits", "Pub 15-B",
     "irs_publication", [], {2025: (f"{IRS}/pub/irs-prior/p15b--2025.pdf", "pdf"),
                             2026: (f"{IRS}/publications/p15b", "irs_html")}),
    # Revision-dated (12/2024), not annual: the same edition serves both years.
    ("pub-583", "Publication 583, Starting a Business and Keeping Records", "Pub 583",
     "irs_publication", [], {2025: (f"{IRS}/publications/p583", "irs_html"),
                             2026: (f"{IRS}/publications/p583", "irs_html")}),
    # Employer payroll: wages paid to family members, fringe benefits, withholding. The live
    # page is already the 2026 edition; 2025 survives only as a PDF (as with Pub 15-B).
    ("pub-15", "Publication 15 (Circular E), Employer's Tax Guide", "Pub 15",
     "irs_publication", [], {2025: (f"{IRS}/pub/irs-prior/p15--2025.pdf", "pdf"),
                             2026: (f"{IRS}/publications/p15", "irs_html")}),
    ("instr-sch-c", "Instructions for Schedule C (Form 1040)", "Instructions for Schedule C",
     "form_instructions", ["sole_prop"], {2025: (f"{IRS}/instructions/i1040sc", "irs_html")}),
    ("instr-4562", "Instructions for Form 4562", "Instructions for Form 4562",
     "form_instructions", [], {2025: (f"{IRS}/instructions/i4562", "irs_html")}),
    ("instr-8829", "Instructions for Form 8829", "Instructions for Form 8829",
     "form_instructions", ["sole_prop"], {2025: (f"{IRS}/instructions/i8829", "irs_html")}),
    ("instr-1120", "Instructions for Form 1120", "Instructions for Form 1120",
     "form_instructions", ["c_corp"], {2025: (f"{IRS}/instructions/i1120", "irs_html")}),
    ("instr-1120-s", "Instructions for Form 1120-S", "Instructions for Form 1120-S",
     "form_instructions", ["s_corp"], {2025: (f"{IRS}/instructions/i1120s", "irs_html")}),
    ("instr-1065", "Instructions for Form 1065", "Instructions for Form 1065",
     "form_instructions", ["partnership"], {2025: (f"{IRS}/instructions/i1065", "irs_html")}),
    # Qualified business income deduction (simplified computation), incl. the thresholds.
    ("instr-8995", "Instructions for Form 8995", "Instructions for Form 8995",
     "form_instructions", PASS_THROUGH, {2025: (f"{IRS}/instructions/i8995", "irs_html")}),
]


def main() -> None:
    sources = []
    for section, entities in IRC.items():
        url = ("https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section"
               f"{section}&num=0&edition=prelim")
        entry = {"id": f"irc-{section.lower()}", "title": f"26 U.S.C. § {section}",
                 "citation_root": f"IRC § {section}", "doc_type": "irc",
                 "editions": {y: {"url": url, "parser": "uscode_html"} for y in ECFR_DATES}}
        if entities:
            entry["entity_types"] = entities
        if section in IRC_OVERRIDES:
            entry["entity_overrides"] = IRC_OVERRIDES[section]
        sources.append(entry)
    for section in REGS:
        # Case matters: "1.263(a)-1" and "1.263A-1" are different sections.
        entry = {"id": f"reg-{section.replace('(', '_').replace(')', '')}",
                 "title": f"26 CFR § {section}", "citation_root": f"Treas. Reg. § {section}",
                 "doc_type": "treasury_regulation",
                 "editions": {y: {"url": ("https://www.ecfr.gov/api/versioner/v1/full/"
                                          f"{d}/title-26.xml?part=1&section={quote(section)}"),
                                  "parser": "ecfr_xml"} for y, d in ECFR_DATES.items()}}
        if section.startswith("1.199A"):
            entry["entity_types"] = PASS_THROUGH
        sources.append(entry)
    for sid, title, root, doc_type, entities, editions in PUBLICATIONS:
        entry = {"id": sid, "title": title, "citation_root": root, "doc_type": doc_type,
                 "editions": {y: {"url": u, "parser": p} for y, (u, p) in editions.items()}}
        if entities:
            entry["entity_types"] = entities
        sources.append(entry)
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: object) -> bool:
            return True

    header = "# Generated by scripts/generate_sources.py; edit that script, not this file.\n"
    OUT.write_text(header + yaml.dump({"sources": sources}, Dumper=NoAliasDumper, sort_keys=False,
                                    allow_unicode=True),
                   encoding="utf-8")
    print(f"wrote {len(sources)} sources to {OUT}")


if __name__ == "__main__":
    main()
