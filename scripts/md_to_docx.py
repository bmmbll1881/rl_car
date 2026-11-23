"""Convert the experiment report (markdown subset) into a .docx.

Supported: `#`..`####` headings, paragraphs, `-` bullets, `1.` ordered lists,
pipe tables, fenced code blocks, `![alt](path)` images, `**bold**`, `` `code` ``.

    python scripts/md_to_docx.py docs/report.md docs/report.docx
"""
from __future__ import annotations

import argparse
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CJK_FONT = "等线"
MONO_FONT = "Consolas"


def add_rule(doc):
    """Thin horizontal rule (markdown ``---``)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "BBBBBB")
    pbdr.append(bottom)
    pPr.append(pbdr)


def set_base_styles(doc):
    style = doc.styles["Normal"]
    style.font.name = CJK_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.line_spacing = 1.15
    for name, size, color in (("Heading 1", 18, 0x000000), ("Heading 2", 14, 0x000000),
                              ("Heading 3", 12, 0x000000), ("Heading 4", 11, 0x000000)):
        st = doc.styles[name]
        st.font.name = CJK_FONT
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor((color >> 16) & 255, (color >> 8) & 255, color & 255)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
        st.paragraph_format.space_before = Pt(10 if name != "Heading 3" else 8)
        st.paragraph_format.space_after = Pt(4)


def add_runs(par, text):
    """Handle **bold** and `code` inline spans."""
    for token in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            r = par.add_run(token[2:-2])
            r.bold = True
        elif token.startswith("`") and token.endswith("`"):
            r = par.add_run(token[1:-1])
            r.font.name = MONO_FONT
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0xA3, 0x1E, 0x1E)
        else:
            par.add_run(token)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, cell in enumerate(header):
        p = table.rows[0].cells[i].paragraphs[0]
        r = p.add_run(cell)
        r.bold = True
        r.font.size = Pt(9.5)
    # repeat the header row when the table breaks across pages
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    trPr.append(tbl_header)

    for row in body:
        cells = table.add_row().cells
        for i, val in enumerate(row[: len(header)]):
            p = cells[i].paragraphs[0]
            add_runs(p, val)
            for r in p.runs:
                r.font.size = Pt(9)

    # column widths proportional to content (CJK counts double)
    def vis(s):
        return sum(2 if ord(ch) > 127 else 1 for ch in s)

    weights = []
    for c in range(len(header)):
        longest = max([vis(r[c]) if c < len(r) else 0 for r in rows] + [vis(header[c])])
        weights.append(max(6.0, min(float(longest), 44.0)))
    total = sum(weights)
    widths = [max(1.4, 16.2 * w / total) for w in weights]
    scale = 16.2 / sum(widths)
    widths = [w * scale for w in widths]
    table.autofit = False
    for i, wcm in enumerate(widths):
        table.columns[i].width = Cm(wcm)   # the table grid LibreOffice/Word actually uses
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Cm(widths[i])
    doc.add_paragraph()


def add_code(doc, lines):
    for line in lines:
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(0.4)
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.space_before = Pt(0)
        par.paragraph_format.line_spacing = 1.0
        pPr = par._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "F4F5F7")
        pPr.append(shd)
        r = par.add_run(line if line else " ")
        r.font.name = MONO_FONT
        r.font.size = Pt(9)
        r.italic = False
        r.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
        # set every font slot so LibreOffice/Word do not substitute an italic face
        rPr = r._element.get_or_add_rPr()
        rFonts = rPr.get_or_add_rFonts()
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rFonts.set(qn(attr), MONO_FONT)


def build(md_path, docx_path, title=None, base_dir=None):
    base_dir = base_dir or os.path.dirname(os.path.abspath(md_path))
    with open(md_path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()

    doc = Document()
    set_base_styles(doc)
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.4)
        section.right_margin = Cm(2.4)

    if title:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(22)
        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            add_code(doc, block)
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = re.sub(r"[*`]", "", m.group(2))
            doc.add_heading(text, level=min(level, 4))
            i += 1
            continue

        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            path = m.group(2)
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(base_dir, path))
            if os.path.exists(path):
                doc.add_picture(path, width=Cm(16.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if m.group(1):
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = cap.add_run(m.group(1))
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
            else:
                print(f"WARNING: missing image -> {path}")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "|".join(row)):
                    rows.append(row)
                i += 1
            if rows:
                add_table(doc, rows)
            continue

        m = re.match(r"^([-*])\s+(.*)$", stripped)
        if m:
            text = m.group(2)
            i += 1
            # markdown "lazy continuation": indented follow-up lines belong to the item
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) and not lines[i].strip().startswith("|"):
                text += " " + lines[i].strip()
                i += 1
            par = doc.add_paragraph(style="List Bullet")
            add_runs(par, text)
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            # manual numbering: Word's List Number style continues across lists
            text = m.group(2)
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) and not lines[i].strip().startswith("|"):
                text += " " + lines[i].strip()
                i += 1
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.75)
            par.paragraph_format.first_line_indent = Cm(-0.55)
            par.paragraph_format.space_after = Pt(3)
            par.add_run(f"{m.group(1)}. ")
            add_runs(par, text)
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            add_rule(doc)
            i += 1
            continue

        if stripped.startswith(">"):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.5)
            add_runs(par, stripped.lstrip("> ").strip())
            for r in par.runs:
                r.italic = True
                r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
            i += 1
            continue

        if stripped:
            text = stripped
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) and not lines[i].strip().startswith("|"):
                text += " " + lines[i].strip()
                i += 1
            par = doc.add_paragraph()
            add_runs(par, text)
            continue
        i += 1

    os.makedirs(os.path.dirname(os.path.abspath(docx_path)), exist_ok=True)
    doc.save(docx_path)
    print("wrote", docx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    build(args.src, args.dst, args.title)
