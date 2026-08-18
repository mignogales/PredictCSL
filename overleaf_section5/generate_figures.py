#!/usr/bin/env python3
"""Regenerate Section 5 figures from frozen CSVs using bundled libraries."""

from __future__ import annotations

import csv
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frozen"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
pdfmetrics.registerFont(TTFont("FigureSans", REGULAR))
pdfmetrics.registerFont(TTFont("FigureSans-Bold", BOLD))

INK = HexColor("#253243")
MUTED = HexColor("#667085")
GRID = HexColor("#E4E7EC")
BLUE = HexColor("#2563EB")
TEAL = HexColor("#0F766E")
ORANGE = HexColor("#D97706")
RED = HexColor("#B42318")
PALE_BLUE = HexColor("#F5F8FF")
PALE_ORANGE = HexColor("#FFF9F0")
PROFILE_COLORS = {
    "conservative": TEAL,
    "balanced": BLUE,
    "aggressive": ORANGE,
    "extreme": RED,
    "very_extreme": HexColor("#7A1F5C"),
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def render_png(pdf_path: Path) -> None:
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    bitmap = page.render(scale=240 / 72)
    bitmap.to_pil().save(pdf_path.with_suffix(".png"))
    page.close()
    pdf.close()


def finish(c: canvas.Canvas, stem: str) -> None:
    c.save()
    render_png(FIGURES / f"{stem}.pdf")


def text(c, x, y, value, size=8, color=INK, bold=False, align="left"):
    c.setFont("FigureSans-Bold" if bold else "FigureSans", size)
    c.setFillColor(color)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def multiline(c, x, y, value, size=7.2, color=INK, bold=False,
              align="center", leading=9):
    for index, line in enumerate(value.split("\n")):
        text(c, x, y - index * leading, line, size, color, bold, align)


def box(c, x, y, width, height, title, body, stroke, fill):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.25)
    c.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    text(c, x + width / 2, y + height * 0.66, title, 8.1, stroke, True, "center")
    multiline(c, x + width / 2, y + height * 0.35, body, 6.8, INK,
              False, "center", 8.4)


def arrow(c, x1, y1, x2, y2, color=MUTED):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.1)
    c.line(x1, y1, x2 - 5, y2)
    c.line(x2 - 5, y2, x2 - 10, y2 + 3)
    c.line(x2 - 5, y2, x2 - 10, y2 - 3)


def figure_pipeline() -> None:
    stem = "fig1_risk_selector_pipeline"
    width, height = 11.2 * inch, 4.35 * inch
    c = canvas.Canvas(str(FIGURES / f"{stem}.pdf"), pagesize=(width, height))

    # Two clean lanes: model fitting above, one-pass selection below.
    c.setFillColor(PALE_ORANGE)
    c.roundRect(14, 166, width - 28, 131, 9, fill=1, stroke=0)
    c.setFillColor(PALE_BLUE)
    c.roundRect(14, 28, width - 28, 116, 9, fill=1, stroke=0)
    text(c, 28, height - 24, "CALIBRATE ON SYNTHETIC FORECASTS", 10.0, ORANGE, True)
    text(c, 28, 133, "SELECT ON AN UNLABELED REAL CONTEXT", 10.0, BLUE, True)

    top_y, top_h = 190, 69
    top_cards = [
        (32, 142, "Synthetic contexts", "12k train + 2k calibration\nsix forecast horizons"),
        (203, 142, "Frozen TSFM labels", "candidate/native MAE ratios\n+ more-than-5% harm events"),
        (374, 178, "Expected-risk predictor", "mean log-risk  ·  uncertainty\nprobability of material harm"),
    ]
    for x, w, title, body in top_cards:
        box(c, x, top_y, w, top_h, title, body, ORANGE, white)
    arrow(c, 176, top_y + top_h / 2, 201, top_y + top_h / 2)
    arrow(c, 347, top_y + top_h / 2, 372, top_y + top_h / 2)

    # Calibration card: the selectable control is the visual focal point.
    cal_x, cal_w = 581, 194
    c.setFillColor(white)
    c.setStrokeColor(ORANGE)
    c.setLineWidth(1.5)
    c.roundRect(cal_x, top_y, cal_w, top_h, 7, fill=1, stroke=1)
    text(c, cal_x + cal_w / 2, top_y + 49,
         "Five nested harm profiles", 8.4, ORANGE, True, "center")
    budgets = [("0.5%", TEAL), ("1%", BLUE), ("3%", ORANGE), ("15%", RED),
               ("20%", PROFILE_COLORS["very_extreme"])]
    for index, (label, color) in enumerate(budgets):
        px = cal_x + 25 + index * 36
        c.setFillColor(color)
        c.setStrokeColor(white)
        c.circle(px, top_y + 25, 12, fill=1, stroke=1)
        text(c, px, top_y + 22, label, 6.8, white, True, "center")
    text(c, cal_x + cal_w / 2, top_y + 6,
         "synthetic >5%-harm budgets", 6.5, MUTED, False, "center")
    arrow(c, 554, top_y + top_h / 2, 579, top_y + top_h / 2)

    bottom_y, bottom_h = 49, 62
    bottom_cards = [
        (32, 137, "Context + horizon", "no target; no real error"),
        (198, 166, "Score candidate windows", "q = mean + uncertainty + harm"),
        (394, 174, "Apply chosen profile", "accept the shortest window\nbelow its calibrated threshold"),
        (598, 177, "Forecast or abstain", "selected window  ·  otherwise native"),
    ]
    for x, w, title, body in bottom_cards:
        box(c, x, bottom_y, w, bottom_h, title, body, BLUE, white)
    for (x, w, _, _), (next_x, _, _, _) in zip(bottom_cards[:-1], bottom_cards[1:]):
        arrow(c, x + w + 2, bottom_y + bottom_h / 2,
              next_x - 2, bottom_y + bottom_h / 2, BLUE)

    # Learned scorer and calibrated thresholds enter at the steps that use them.
    c.setStrokeColor(ORANGE)
    c.setLineWidth(1.15)
    model_x = 374 + 178 / 2
    score_x = 198 + 166 / 2
    c.line(model_x, top_y - 2, score_x, bottom_y + bottom_h + 12)
    c.line(score_x, bottom_y + bottom_h + 12, score_x - 3, bottom_y + bottom_h + 18)
    c.line(score_x, bottom_y + bottom_h + 12, score_x + 3, bottom_y + bottom_h + 18)
    c.setStrokeColor(PROFILE_COLORS["very_extreme"])
    gate_x = 394 + 174 / 2
    c.line(cal_x + cal_w / 2, top_y - 2, gate_x, bottom_y + bottom_h + 12)
    c.line(gate_x, bottom_y + bottom_h + 12, gate_x - 3, bottom_y + bottom_h + 18)
    c.line(gate_x, bottom_y + bottom_h + 12, gate_x + 3, bottom_y + bottom_h + 18)

    text(c, width / 2, 10,
         "User chooses the harm budget; native context remains the explicit fallback.",
         7.8, MUTED, False, "center")
    finish(c, stem)


def map_value(value, low, high, start, end):
    return start + (value - low) / (high - low) * (end - start)


def figure_compute_harm_dial() -> None:
    stem = "fig2_compute_harm_dial"
    rows = read_csv("risk_profile_summary.csv")
    order = ["conservative", "balanced", "aggressive", "extreme", "very_extreme"]
    indexed = {row["profile"]: row for row in rows}
    rows = [indexed[p] for p in order]
    width, height = 9.15 * inch, 4.25 * inch
    c = canvas.Canvas(str(FIGURES / f"{stem}.pdf"), pagesize=(width, height))
    text(c, width / 2, height - 17,
         "Real zero-shot operating points across 11 forecasters",
         11, INK, True, "center")

    # Left scatter panel.
    x0, y0, x1, y1 = 48, 46, width * 0.58 - 18, height - 72
    text(c, (x0 + x1) / 2, height - 38,
         "The five profiles form a compute–harm dial", 9.4, INK, True, "center")
    c.setStrokeColor(GRID)
    c.setLineWidth(0.6)
    for harm in range(0, 13, 2):
        x = map_value(harm, 0, 12.5, x0, x1)
        c.line(x, y0, x, y1)
        text(c, x, y0 - 13, str(harm), 7.5, MUTED, False, "center")
    for saving in range(0, 61, 10):
        y = map_value(saving, 0, 58, y0, y1)
        c.line(x0, y, x1, y)
        text(c, x0 - 7, y - 2.5, str(saving), 7.5, MUTED, False, "right")
    points = []
    for row in rows:
        harm = float(row["pooled_harm5_pct"])
        saving = float(row["mean_flops_saved_pct"])
        points.append((map_value(harm, 0, 12.5, x0, x1),
                       map_value(saving, 0, 58, y0, y1), row["profile"]))
    c.setStrokeColor(HexColor("#98A2B3"))
    c.setLineWidth(1.8)
    for first, second in zip(points[:-1], points[1:]):
        c.line(first[0], first[1], second[0], second[1])
    labels = {
        "conservative": (7, 7, "Conservative · 0.5%"),
        "balanced": (7, 7, "Balanced · 1%"),
        "aggressive": (-52, 13, "Aggressive · 3%"),
        "extreme": (8, -22, "Efficiency · 15%"),
        "very_extreme": (-105, 10, "Max efficiency · 20%"),
    }
    for x, y, profile in points:
        color = PROFILE_COLORS[profile]
        c.setFillColor(color)
        c.setStrokeColor(white)
        c.circle(x, y, 5.6, fill=1, stroke=1)
        dx, dy, label = labels[profile]
        text(c, x + dx, y + dy, label, 7.4, color, True)
    text(c, (x0 + x1) / 2, 16,
         "Observed instances harmed by more than 5% (%)", 8.2, INK, False, "center")
    c.saveState()
    c.translate(14, (y0 + y1) / 2)
    c.rotate(90)
    text(c, 0, 0, "Mean theoretical TSFM FLOPs saved (%)", 8.2, INK, False, "center")
    c.restoreState()

    # Right bar panel.
    bx0, by0, bx1, by1 = width * 0.64, 46, width - 21, height - 72
    text(c, (bx0 + bx1) / 2, height - 38,
         "Accuracy remains close to native", 9.4, INK, True, "center")
    c.setStrokeColor(GRID)
    for tick in [-0.8, -0.6, -0.4, -0.2, 0.0]:
        y = map_value(tick, -0.9, 0.04, by0, by1)
        c.line(bx0, y, bx1, y)
        text(c, bx0 - 6, y - 2.5, f"{tick:.3f}", 7.2, MUTED, False, "right")
    bar_gap = (bx1 - bx0) / 5
    zero_y = map_value(0, -0.9, 0.04, by0, by1)
    short = ["Cons.", "Bal.", "Aggr.", "Efficiency", "Max eff."]
    for index, row in enumerate(rows):
        value = 100 * (1 - float(row["model_geomean_mase_ratio"]))
        x = bx0 + (index + 0.5) * bar_gap
        value_y = map_value(value, -0.9, 0.04, by0, by1)
        c.setFillColor(PROFILE_COLORS[row["profile"]])
        c.rect(x - 13, value_y, 26, zero_y - value_y, fill=1, stroke=0)
        text(c, x, value_y - 12, f"{value:+.3f}%", 7.2, INK, True, "center")
        text(c, x, by0 - 14, short[index], 7.4, MUTED, False, "center")
    c.setStrokeColor(MUTED)
    c.setLineWidth(0.9)
    c.line(bx0, zero_y, bx1, zero_y)
    c.saveState()
    c.translate(width * 0.60, (by0 + by1) / 2)
    c.rotate(90)
    text(c, 0, 0, "Aggregate MASE change vs native (%)", 8.2, INK, False, "center")
    c.restoreState()
    finish(c, stem)


def blend(low: str, high: str, amount: float) -> Color:
    low_c, high_c = HexColor(low), HexColor(high)
    amount = max(0.0, min(1.0, amount))
    return Color(low_c.red + (high_c.red - low_c.red) * amount,
                 low_c.green + (high_c.green - low_c.green) * amount,
                 low_c.blue + (high_c.blue - low_c.blue) * amount)


def heatmap(c, x0, y0, width, height, values, row_labels, columns,
            title, low_color, high_color, maximum, show_rows=True):
    rows, cols = len(values), len(columns)
    label_w = 105 if show_rows else 5
    grid_x = x0 + label_w
    grid_w = width - label_w
    cell_w, cell_h = grid_w / cols, height / rows
    text(c, grid_x + grid_w / 2, y0 + height + 34, title, 9.5, INK, True, "center")
    for j, label in enumerate(columns):
        text(c, grid_x + (j + 0.5) * cell_w, y0 + height + 13,
             label, 7.5, MUTED, True, "center")
    for i, (label, row) in enumerate(zip(row_labels, values)):
        y = y0 + height - (i + 1) * cell_h
        if show_rows:
            text(c, grid_x - 6, y + cell_h / 2 - 2.5, label, 7.4, INK,
                 i == 4, "right")
        for j, value in enumerate(row):
            fill = blend(low_color, high_color, value / maximum)
            c.setFillColor(fill)
            c.setStrokeColor(white)
            c.rect(grid_x + j * cell_w, y, cell_w, cell_h, fill=1, stroke=1)
            color = white if value / maximum > 0.56 else INK
            text(c, grid_x + (j + 0.5) * cell_w, y + cell_h / 2 - 2.8,
                 f"{value:.1f}%", 7.2, color, True if i == 4 else False, "center")
    # Highlight FlowState row.
    y = y0 + height - 5 * cell_h
    c.setStrokeColor(RED)
    c.setLineWidth(1.7)
    c.rect(grid_x, y, grid_w, cell_h, fill=0, stroke=1)
    return grid_x, grid_w, cell_h


def figure_model_heatmaps() -> None:
    stem = "fig3_model_profile_heatmaps"
    rows = read_csv("risk_profiles_all_models.csv")
    profiles = ["conservative", "balanced", "aggressive", "extreme", "very_extreme"]
    columns = ["Conservative", "Balanced", "Aggressive", "Efficiency", "Max efficiency"]
    models = [
        "Chronos2-Base", "Chronos2-Small", "Chronos2-Synth",
        "ChronosBolt-Base", "FlowState-R1", "Moirai2-Small",
        "PatchTST-FM-R1", "Sundial-Base-128M", "TimesFM2.5-200M",
        "TiRex2", "Toto-2.0-313m",
    ]
    indexed = {(row["model"], row["profile"]): row for row in rows}
    savings = [[float(indexed[(model, p)]["flops_saved_pct"]) for p in profiles]
               for model in models]
    harm = [[float(indexed[(model, p)]["harm5_pct"]) for p in profiles]
            for model in models]

    width, height = 11.4 * inch, 6.15 * inch
    c = canvas.Canvas(str(FIGURES / f"{stem}.pdf"), pagesize=(width, height))
    text(c, width / 2, height - 18,
         "The harm dial is ordered, but transfer remains model specific",
         11, INK, True, "center")
    top, bottom = height - 82, 34
    grid_height = top - bottom
    panel_w = (width - 42) / 2
    heatmap(c, 11, bottom, panel_w, grid_height, savings, models, columns,
            "Theoretical TSFM FLOPs saved", "#F7FCF0", "#006D77", 78, True)
    heatmap(c, 31 + panel_w, bottom, panel_w, grid_height, harm, models, columns,
            "Observed instances harmed by more than 5%", "#FFF7EC", "#B42318", 17, False)
    finish(c, stem)


def main() -> None:
    figure_pipeline()
    figure_compute_harm_dial()
    figure_model_heatmaps()
    print(f"Wrote Section 5 risk-predictor figures to {FIGURES}")


if __name__ == "__main__":
    main()
