"""make_pdf.py - render Chronos_Solver_ARC_Prize_2026.pdf from PAPER.md + figures/.

    python paper_writing/make_pdf.py

Requires: reportlab, pillow, pypdf (pip install reportlab pillow pypdf).
"""
"""Build a clean paper-style PDF from PAPER.md + the figures."""
import os, re
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "Chronos_Solver_ARC_Prize_2026.pdf")

INK, MUTE, NAVY, ERA2 = HexColor("#1f2330"), HexColor("#6b7280"), HexColor("#0f1b2d"), HexColor("#3b6fb5")
NOTE_BG = HexColor("#eef2f8")

TITLE = "Chronos Solver: Why Live Search Beats Black-Box Neural Agents on ARC-AGI-3"
SUBTITLE = ("An honest 19-iteration account: live white-box search scores 0.22 while a "
            "black-box neural agent scores 0.01 on the same ARC-AGI-3 games.")
AUTHOR = "Shreyas Mahimkar"
LINKS = ('Code: github.com/shreyasmahimkar/arc-agi-3 &nbsp;|&nbsp; '
         'Scored notebook (0.22): kaggle.com/code/shreyas4/claude-code-v12-baseline')
ABSTRACT = ("ARC-AGI-3 is an interactive benchmark where an agent must learn a never-seen "
            "game's rules from pixels. Over 19 iterations the Chronos Solver converged on one "
            "decisive fact: because the competition ships each game's source, a solver that "
            "reaches and searches that simulator live (white-box BFS) scores 0.22, while a "
            "purely black-box neural agent scores ~0.01 on the same games. This paper documents "
            "the journey through three eras (LLM orchestration, symbolic search, model-based RL), "
            "argues why genuine search generalises where pattern-learning does not, and reports "
            "the diagnostic ablations - including a disproved hypothesis - that localise the "
            "current bottleneck to representation, not compute.")

# figures placed AFTER the section whose header contains the key
FIG_AFTER = {
    "headline result": [("fig1_scores.png", "Figure 1. Genuine live search scores 22x the black-box neural agent on the same games.")],
    "three eras": [("fig2_timeline.png", "Figure 2. Three eras, each diagnosing the last era's bottleneck."),
                   ("fig3_coverage.png", "Figure 3. v12 solved 31 levels across 13 games from scratch (read from the BFS caches).")],
    "unlocked 0.22": [("fig4_chaining.png", "Figure 4. Chaining real level baselines fixed correctness and roughly halved solutions.")],
    "diagnosing the wall": [("fig5_wall.png", "Figure 5. ls20: deep levels die of breadth, not impossibility (L5 is the wall).")],
    "generalisation science": [("fig6_wm_fix.png", "Figure 6. Make 'copy' the default: fresh-episode accuracy 36.6% -> >90% (v14->v15)."),
                               ("fig7_honest_negative.png", "Figure 7. A disproved colour-augmentation hypothesis, reported as a negative.")],
    "v19 system": [("fig8_architecture.png", "Figure 8. v19 routing: BFS-first, learned fallback, cache only on timeout, plus the ExIt flywheel.")],
}

UNI = {"→": "->", "×": "x", "≤": "<=", "≥": ">=", "≈": "~", "−": "-", "★": "*",
       "✓": "OK", "·": "-", "≠": "!=", "∈": "in"}

def inline(t):
    for k, v in UNI.items():
        t = t.replace(k, v)
    t = t.replace("\\*", "\x00")          # protect escaped asterisks (e.g. "A\*")
    # xml-escape first
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # then markdown -> reportlab markup
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size=9>\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = t.replace("\x00", "*")             # restore literal asterisk
    return t

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold",
                            fontSize=18, leading=22, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4),
    "sub": ParagraphStyle("s", fontSize=11.5, leading=15, textColor=MUTE, alignment=TA_LEFT,
                          fontName="Helvetica-Oblique", spaceAfter=8),
    "meta": ParagraphStyle("m", fontSize=9, leading=12, textColor=INK, alignment=TA_LEFT, spaceAfter=2),
    "abhead": ParagraphStyle("ah", fontSize=9.5, leading=12, textColor=NAVY, fontName="Helvetica-Bold"),
    "abody": ParagraphStyle("ab", fontSize=9.3, leading=13.2, textColor=INK, alignment=TA_JUSTIFY),
    "h": ParagraphStyle("h", fontSize=12.5, leading=15, textColor=NAVY, fontName="Helvetica-Bold",
                        spaceBefore=12, spaceAfter=4),
    "body": ParagraphStyle("b", fontSize=10, leading=14, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bl", fontSize=10, leading=13.5, textColor=INK, alignment=TA_LEFT,
                             leftIndent=16, bulletIndent=4, spaceAfter=3),
    "note": ParagraphStyle("n", fontSize=9.3, leading=13, textColor=HexColor("#33405a"),
                           alignment=TA_LEFT, leftIndent=10, rightIndent=10, spaceBefore=2, spaceAfter=6,
                           fontName="Helvetica-Oblique", backColor=NOTE_BG, borderPadding=6),
    "cap": ParagraphStyle("c", fontSize=8.5, leading=11, textColor=MUTE, alignment=TA_CENTER,
                          fontName="Helvetica-Oblique", spaceBefore=2, spaceAfter=12),
    "ref": ParagraphStyle("r", fontSize=9, leading=12.5, textColor=INK, alignment=TA_LEFT,
                          leftIndent=14, firstLineIndent=-14, spaceAfter=3),
}

USABLE_W = 6.7 * inch

def img_flow(fname, caption, max_w=USABLE_W):
    p = os.path.join(FIG, fname)
    w, h = PILImage.open(p).size
    iw = min(max_w, w)
    ih = iw * h / w
    return KeepTogether([Image(p, width=iw, height=ih), Paragraph(inline(caption), S["cap"])])

# -------- parse PAPER.md --------
raw = open(os.path.join(ROOT, "PAPER.md")).read()
ruler = "=" * 66
parts = raw.split(ruler)
body = parts[1].strip() if len(parts) > 1 else raw
biblio = parts[2].strip() if len(parts) > 2 else ""

story = []
# title block
story += [Paragraph(inline(TITLE), S["title"]), Paragraph(inline(SUBTITLE), S["sub"])]
story += [Paragraph(AUTHOR + " &nbsp;|&nbsp; ARC Prize 2026 - Paper Track", S["meta"]),
          Paragraph(LINKS, S["meta"]), Spacer(1, 6),
          HRFlowable(width="100%", thickness=1, color=HexColor("#d7dce6")), Spacer(1, 6)]
# abstract
story += [Paragraph("Abstract", S["abhead"]),
          Paragraph(inline(ABSTRACT), S["abody"]), Spacer(1, 4)]
# graphical abstract (cover)
story += [img_flow("fig0_cover.png", "Chronos Solver - 19 iterations, one honest lesson.", max_w=5.4 * inch)]

def render_block(text, pending):
    """Parse a markdown block region into flowables; manage pending-figure flush.

    Handles multi-line bullets (indented continuation) and multi-line > note blocks.
    """
    lines = text.split("\n")
    buf, bullets, note = [], [], []
    def flush_para():
        nonlocal buf
        if buf:
            story.append(Paragraph(inline(" ".join(buf).strip()), S["body"])); buf = []
    def flush_bullets():
        nonlocal bullets
        for b in bullets:
            story.append(Paragraph(inline(b), S["bullet"], bulletText="•"))
        bullets = []
    def flush_note():
        nonlocal note
        if note:
            story.append(Paragraph(inline(" ".join(note).strip()), S["note"])); note = []
    def flush_all():
        flush_para(); flush_bullets(); flush_note()

    for raw in lines:
        s = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if not s:
            flush_all(); continue
        if s.startswith("## "):
            flush_all()
            for f, c in pending[0]:           # figures belong AFTER the previous section
                story.append(img_flow(f, c))
            pending[0] = []
            head = s[3:].strip()
            story.append(Paragraph(inline(head), S["h"]))
            key = head.lower()
            for k, figs in FIG_AFTER.items():
                if k in key:
                    pending[0] = figs
            continue
        if s.startswith(">"):
            flush_para(); flush_bullets()
            note.append(s.lstrip(">").strip()); continue
        if s.startswith("- "):
            flush_para(); flush_note()
            bullets.append(s[2:].strip()); continue
        if bullets and indent >= 2:           # indented continuation of the current bullet
            bullets[-1] += " " + s; continue
        m = re.match(r"^(\d+)\.\s+(.*)", s)
        if m and not bullets:                 # numbered (bibliography) -> hanging-indent ref
            flush_all()
            story.append(Paragraph(f'{m.group(1)}. {inline(m.group(2))}', S["ref"])); continue
        flush_bullets(); flush_note()         # a normal line ends any list/note
        buf.append(s)
    flush_all()

pending = [[]]
render_block(body, pending)
for f, c in pending[0]:
    story.append(img_flow(f, c))
pending[0] = []

# bibliography
if biblio:
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=HexColor("#d7dce6")))
    render_block(biblio, [[]])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTE)
    canvas.drawString(0.85 * inch, 0.5 * inch, "Chronos Solver - ARC Prize 2026 Paper Track")
    canvas.drawRightString(letter[0] - 0.85 * inch, 0.5 * inch, "Page %d" % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                        title=TITLE, author=AUTHOR, subject="ARC Prize 2026 Paper Track")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
print("pages:", end=" ")
from pypdf import PdfReader
print(len(PdfReader(OUT).pages))
