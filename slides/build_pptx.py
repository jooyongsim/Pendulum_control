r"""Build the pendulum deck as a PPTX matching the AMSL lecture design system.

The slide is 1280x720 px at 96 dpi = 13.333 x 7.5 in, so px/96 maps straight to
inches and the design-system geometry can be used verbatim.

Equations are rendered to 300 dpi PNGs (Computer Modern) and placed at their
natural size, which reproduces the LaTeX point size exactly. Prose containing
inline $...$ is rendered the same way but with Malgun Gothic for the Hangul, so
a symbol like theta-dot or omega_d never degrades to an ASCII underscore.

Layout is a single vertical flow with one left margin and one gap constant.
Table columns are sized from their own content, tables stack vertically, and the
build asserts that nothing runs into the footer band.
"""
import os
import sys

import numpy as np

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
# DECK_CONTENT selects which deck to build: the module has to define
# COVER, EQ, FOOTER and SLIDES (see deck_content_place.py).
_content = importlib.import_module(os.environ.get('DECK_CONTENT',
                                                  'deck_content_place'))
COVER, EQ, FOOTER, SLIDES = (_content.COVER, _content.EQ,
                             _content.FOOTER, _content.SLIDES)
from render_eq import render_all, render_mixed_all  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "deck.pptx"
EQ_DIR = sys.argv[2] if len(sys.argv) > 2 else "eq_png"
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
# the AMS Lab mark the lecture deck itself uses, so the two decks match
LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amslab_logo.png")

NAVY = RGBColor(0x20, 0x38, 0x64)
PALE = RGBColor(0xB4, 0xC3, 0xD6)
INK = RGBColor(0x00, 0x00, 0x00)
GREY = RGBColor(0x40, 0x40, 0x40)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xFF, 0x00, 0x00)
BAND = RGBColor(0xF4, 0xF7, 0xFB)
CORE = "Arial"
HANGUL = "Malgun Gothic"

EQ_DPI = 300
LEFT = 46                 # one left margin for every block on the slide
WIDTH = 1280 - 2 * 46
GAP = 11                  # one vertical gap between blocks
FOOTER_TOP = 694.2
SAFE_BOTTOM = FOOTER_TOP - 18


def px(v):
    return Inches(v / 96.0)


def rect(slide, l, t, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, px(l), px(t), px(w), px(h))
    s.fill.solid(); s.fill.fore_color.rgb = color
    s.line.fill.background(); s.shadow.inherit = False
    return s


def textbox(slide, l, t, w, h, text, size, bold=False, italic=False, color=INK,
            font=CORE, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15):
    tb = slide.shapes.add_textbox(px(l), px(t), px(w), px(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color; r.font.name = font
    return tb


def term_box(slide, l, t, w, term, rest, size=14):
    """The deck's signature pattern: bold-italic term, ' : ', then prose."""
    tb = slide.shapes.add_textbox(px(l), px(t), px(w), px(26))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.line_spacing = 1.25
    a = p.add_run(); a.text = term
    a.font.size = Pt(size); a.font.bold = True; a.font.italic = True
    a.font.color.rgb = INK; a.font.name = HANGUL
    b = p.add_run(); b.text = " : " + rest
    b.font.size = Pt(size); b.font.color.rgb = INK; b.font.name = HANGUL
    return tb


def picture(slide, imgs, key, x, y, max_w=WIDTH):
    path, w, h = imgs[key]
    w_in, h_in = w / EQ_DPI, h / EQ_DPI
    if w_in > max_w / 96.0:
        h_in *= (max_w / 96.0) / w_in
        w_in = max_w / 96.0
    slide.shapes.add_picture(path, px(x), px(y), Inches(w_in), Inches(h_in))
    return h_in * 96.0


def text_width_px(s, size):
    """Rough advance width: Hangul/CJK is full width, latin about 0.52 em.

    0.46 was too optimistic for Malgun Gothic: table cells that the estimate
    said would fit wrapped to two lines in PowerPoint, which silently made the
    row taller than add_table() reported and pushed the next block on top of it.
    """
    em = size * 96.0 / 72.0
    return sum(em if ord(c) > 0x2000 else em * 0.52 for c in s)


def wrapped_lines(text, size, width_px):
    """How many lines this text needs at `size` inside `width_px`."""
    return max(1, int(np.ceil(text_width_px(text, size) / width_px)))


def block_height(text, size, width_px, spacing=1.25):
    """Height a wrapped text block will actually occupy, in slide px."""
    line = size * 96.0 / 72.0 * spacing
    return wrapped_lines(text, size, width_px) * line


def col_widths(t, size, pad=24):
    out = []
    for c in range(len(t["head"])):
        cells = [t["head"][c]] + [r[c] for r in t["rows"]]
        out.append(max(text_width_px(x, size) for x in cells) + pad)
    return out


MIN_ROW_PX = 28      # measured from a render: an 11 pt single-line row
                     # comes out about 27 px, so 28 is the honest floor


def row_heights(t, widths, size, row_h=MIN_ROW_PX, margin=16):
    """What each row will REALLY be tall, in slide px.

    A cell whose text wraps grows the row to fit, and PowerPoint will not draw
    a row below a floor of roughly MIN_ROW_PX. Neither was measured before, so
    add_table() under-reported its height and whatever came next landed on top
    of the table.
    """
    line = size * 96.0 / 72.0 * 1.25
    out = []
    for row in [t["head"]] + t["rows"]:
        n = 1
        for c, cell in enumerate(row):
            n = max(n, wrapped_lines(str(cell), size, max(20, widths[c] - margin)))
        out.append(max(row_h, n * line + 8))
    return out


def add_table(slide, t, left, top, size=11, row_h=26):
    """Table sized from its own content, so no cell wraps or clips."""
    widths = col_widths(t, size)
    heights = row_heights(t, widths, size, row_h)
    rows, cols = len(t["rows"]) + 1, len(t["head"])
    shape = slide.shapes.add_table(rows, cols, px(left), px(top),
                                   px(sum(widths)), px(sum(heights)))
    tbl = shape.table
    tbl.first_row = True
    for c, w in enumerate(widths):
        tbl.columns[c].width = px(w)
    for r in range(rows):
        tbl.rows[r].height = px(heights[r])
    for c, head in enumerate(t["head"]):
        cell = tbl.cell(0, c)
        cell.text = head
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        cell.margin_left = cell.margin_right = px(8)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = cell.text_frame.paragraphs[0]
        run = para.runs[0] if para.runs else para.add_run()
        run.font.size = Pt(size); run.font.bold = True
        run.font.color.rgb = WHITE; run.font.name = HANGUL
    for r, row in enumerate(t["rows"], start=1):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = BAND if r % 2 == 0 else WHITE
            cell.margin_left = cell.margin_right = px(8)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = cell.text_frame.paragraphs[0]
            run = para.runs[0] if para.runs else para.add_run()
            run.font.size = Pt(size); run.font.color.rgb = INK
            run.font.name = HANGUL
    return sum(heights)


def chrome(slide, title, lead, number):
    rect(slide, 20.9, 13.5, 9.8, 63.4, NAVY)
    textbox(slide, LEFT, 13.5, 1130, 63.4, title, 28, bold=True, color=NAVY,
            anchor=MSO_ANCHOR.MIDDLE, spacing=0.9)
    if os.path.exists(LOGO):
        slide.shapes.add_picture(LOGO, px(1168), px(8), px(98), px(79))
    top = 88
    if lead:
        lead_h = block_height(lead, 16, WIDTH, spacing=1.2)
        textbox(slide, LEFT, 90, WIDTH, lead_h, lead, 16, bold=True, font=HANGUL, spacing=1.2)
        top = 90 + lead_h + 22
    rect(slide, 0, FOOTER_TOP, 1280, 26.5, NAVY)
    rect(slide, 1280 - 331.6, FOOTER_TOP, 331.6, 26.5, PALE)
    textbox(slide, 8, FOOTER_TOP, 700, 26.5, FOOTER, 12, bold=True, italic=True,
            color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, 1215.9, FOOTER_TOP, 63.2, 26.5, str(number), 18, bold=True,
            color=GREY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return top


def collect_prose(slides):
    """Prose lines that carry inline maths; rendered as images, not typed text."""
    out = {}
    for s in slides:
        for i, item in enumerate(s.get("body", [])):
            if item[0] == "plain" and "$" in item[1]:
                out[f"p{s['n']}_{i}"] = item[1]
            elif item[0] == "term" and "$" in item[2]:
                out[f"p{s['n']}_{i}"] = item[1] + " :  " + item[2]
    return out


def build():
    used = set()
    for spec in SLIDES:
        used.update(i[1] for i in spec.get("body", []) if i[0] == "eq")
        if spec.get("tail_eq"):
            used.add(spec["tail_eq"])
    missing = used - set(EQ)
    if missing:
        raise KeyError(f"slides reference undefined equations: {sorted(missing)}")

    eq_imgs = render_all({k: EQ[k] for k in sorted(used)}, EQ_DIR, fontsize=15)
    prose = collect_prose(SLIDES)
    prose_imgs = render_mixed_all(prose, EQ_DIR, fontsize=13) if prose else {}
    print(f"rendered {len(eq_imgs)} equations, {len(prose_imgs)} maths-bearing prose lines")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------------- cover ----------------
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 1280, 720, NAVY)
    rect(s, 35.5, 59.1, 1209, 8.25, RGBColor(0x8F, 0xA3, 0xC0))
    rect(s, 35.5, 663.7, 1209, 8.25, RGBColor(0x8F, 0xA3, 0xC0))
    textbox(s, 64, 286, 1150, 70, COVER["title"], 40, bold=True, color=WHITE,
            font=HANGUL, spacing=1.04)
    textbox(s, 64, 366, 1150, 40, COVER["subtitle"], 20, bold=True,
            color=RGBColor(0xCA, 0xDC, 0xFC))
    tb = s.shapes.add_textbox(px(64), px(456), px(1000), px(130))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_top = 0
    for i, m in enumerate(COVER["meta"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.8
        r = p.add_run(); r.text = m
        r.font.size = Pt(15); r.font.bold = True
        r.font.color.rgb = WHITE; r.font.name = HANGUL
    if os.path.exists(LOGO):
        s.shapes.add_picture(LOGO, px(1280 - 64 - 118), px(520), px(118), px(95))

    # ---------------- content ----------------
    for spec in SLIDES:
        s = prs.slides.add_slide(blank)
        y = chrome(s, spec["title"], spec.get("lead"), spec["n"])

        for i, item in enumerate(spec.get("body", [])):
            key = f"p{spec['n']}_{i}"
            if key in prose_imgs:
                y += picture(s, prose_imgs, key, LEFT, y) + GAP
            elif item[0] == "term":
                # advance by the height the text will really take once wrapped,
                # otherwise a two-line term is overprinted by the next block
                h = block_height(item[1] + " :  " + item[2], 14, WIDTH)
                term_box(s, LEFT, y, WIDTH, item[1], item[2])
                y += h + GAP
            elif item[0] == "plain":
                h = block_height(item[1], 14, WIDTH, spacing=1.2)
                textbox(s, LEFT, y, WIDTH, h, item[1], 14, font=HANGUL, spacing=1.2)
                y += h + GAP
            elif item[0] == "eq":
                y += picture(s, eq_imgs, item[1], LEFT, y) + GAP

        # Tables stack vertically (side by side overlapped and clipped columns).
        # A tall-ish figure goes beside them, filling otherwise dead space; a wide
        # one, or one with no table to sit next to, goes below at full width.
        tables = [t for t in (spec.get("table"), spec.get("table2")) if t]
        fig = spec.get("figure")
        fig_aspect = None
        if fig:
            from PIL import Image
            fig_path = os.path.join(FIG_DIR, fig[0])
            fig_w, fig_h = Image.open(fig_path).size
            fig_aspect = fig_w / fig_h
        beside = bool(tables) and fig is not None and fig_aspect <= 2.0

        # Pick a table font that fits the room that is actually left. Rows have
        # a floor of MIN_ROW_PX, so a dense slide can only be made to fit by
        # dropping a point or two -- better than letting it run into the footer.
        tsize = 11
        if tables:
            reserved = (39 if spec.get("red") else 0) + (34 if fig and not beside else 0)
            while tsize > 9:
                need = sum(sum(row_heights(t, col_widths(t, tsize), tsize))
                           + GAP + 6 for t in tables)
                if y + need + reserved < SAFE_BOTTOM:
                    break
                tsize -= 1

        table_top = y
        widest = 0
        for t in tables:
            widest = max(widest, sum(col_widths(t, tsize)))
            y += add_table(s, t, LEFT, y, size=tsize) + GAP + 6

        # The caption is measured BEFORE the figure is sized: beside a pair of
        # wide tables the picture is narrow, so the caption can wrap to three or
        # four lines. Reserving a fixed 23 px for it is what used to push these
        # slides past the footer.
        red_h = (block_height(spec["red"], 14, WIDTH, spacing=1.2) + GAP
                 if spec.get("red") else 0)
        def fit_figure(top, max_w):
            """Size the picture and its caption together.

            Narrowing the picture makes the caption wrap, which eats the room
            just freed -- so iterate until both fit. Doing it in one pass is
            what let these slides run past the footer.
            """
            fw = max_w
            for _ in range(4):
                cap = block_height(fig[1], 10, fw, spacing=1.2)
                room = SAFE_BOTTOM - top - cap - 5 - GAP - red_h - 4
                fh = min(max_w / fig_aspect, max(40, room))
                new_fw = min(max_w, fh * fig_aspect)
                if abs(new_fw - fw) < 1:
                    fw = new_fw
                    break
                fw = new_fw
            return fw, fw / fig_aspect, block_height(fig[1], 10, fw, spacing=1.2)

        if fig and beside:
            fx = LEFT + widest + 34
            fw, fh, cap_h = fit_figure(table_top, min(1280 - 46 - fx, 780))
            s.shapes.add_picture(fig_path, px(fx), px(table_top), Inches(fw / 96), Inches(fh / 96))
            textbox(s, fx, table_top + fh + 5, fw, cap_h, fig[1], 10, color=GREY, font=HANGUL)
            y = max(y, table_top + fh + 5 + cap_h + GAP)
        elif fig:
            fw, fh, cap_h = fit_figure(y, WIDTH)
            s.shapes.add_picture(fig_path, px(LEFT), px(y), Inches(fw / 96), Inches(fh / 96))
            textbox(s, LEFT, y + fh + 5, fw, cap_h, fig[1], 10, color=GREY, font=HANGUL)
            y += fh + 5 + cap_h + GAP

        if spec.get("red"):
            h = block_height(spec["red"], 14, WIDTH, spacing=1.2)
            textbox(s, LEFT, y, WIDTH, h, spec["red"], 14, bold=True, color=RED,
                    font=HANGUL, spacing=1.2)
            y += h + GAP

        if spec.get("tail_eq"):
            y += picture(s, eq_imgs, spec["tail_eq"], LEFT, y) + GAP

        if spec.get("note"):
            textbox(s, LEFT, SAFE_BOTTOM - 20, WIDTH, 20, spec["note"], 11,
                    color=GREY, font=HANGUL)

        assert y < SAFE_BOTTOM, \
            f"slide {spec['n']} overflows: bottom {y:.0f} px vs safe {SAFE_BOTTOM:.0f}"
        print(f"  slide {spec['n']}: ends at {y:4.0f} px "
              f"({100 * y / FOOTER_TOP:3.0f} % of the way to the footer)")

    prs.save(OUT)
    print(f"written -> {OUT}")


if __name__ == "__main__":
    build()
