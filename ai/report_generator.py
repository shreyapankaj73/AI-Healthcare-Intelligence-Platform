"""
report_generator.py
--------------------
Generates a downloadable PDF health summary report for a worker.
Uses the fpdf2 library (pip install fpdf2).

NOTE: Uses only Latin-1 safe characters -- compatible with fpdf2's
      built-in Helvetica font (no Unicode em-dashes, subscripts, emoji).
"""

from fpdf import FPDF
from datetime import datetime
import re


RISK_COLORS = {
    "high":   (220, 53, 69),
    "medium": (255, 140, 0),
    "low":    (40, 167, 69),
}


def _safe(text: str) -> str:
    """
    Replace characters outside Latin-1 with ASCII equivalents so
    fpdf2's built-in Helvetica font never throws a Unicode error.
    """
    replacements = {
        "\u2014": "-",    # em dash
        "\u2013": "-",    # en dash
        "\u2019": "'",    # right single quote
        "\u2018": "'",    # left single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "-",    # bullet
        "\u2026": "...",  # ellipsis
        "\u00b2": "2",    # superscript 2
        "\u2082": "2",    # subscript 2
        "\u2080": "0",    # subscript 0
        "\u2122": "(TM)",
        "\u00ae": "(R)",
        "\u00b0": " deg",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Drop any remaining non-Latin-1 characters (emoji, etc.)
    text = text.encode("latin-1", errors="ignore").decode("latin-1")
    return text


class HealthReport(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(10, 61, 98)
        self.cell(0, 10, "IOCL AI Healthcare Intelligence Platform", align="C")
        self.ln(5)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Occupational Health Report - Confidential", align="C")
        self.ln(8)
        self.set_draw_color(10, 61, 98)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            f"Generated {datetime.now().strftime('%d %b %Y %H:%M')}  |  Page {self.page_no()}",
            align="C",
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(230, 240, 255)
        self.set_text_color(10, 61, 98)
        self.cell(0, 8, _safe(f"  {title}"), fill=True, ln=True)
        self.ln(2)

    def key_value(self, key: str, value: str, highlight: bool = False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(65, 7, _safe(key))
        self.set_font("Helvetica", "B", 10)
        if highlight:
            self.set_text_color(200, 50, 50)
        else:
            self.set_text_color(20, 20, 20)
        self.cell(0, 7, _safe(str(value)), ln=True)

    def risk_banner(self, risk: str, confidence: float):
        r, g, b = RISK_COLORS.get(risk.lower(), (100, 100, 100))
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.cell(
            0, 14,
            _safe(f"  {risk.upper()} RISK  -  Confidence: {confidence}%"),
            fill=True, align="C", ln=True,
        )
        self.ln(4)


def generate_pdf_report(
    worker_id: str,
    vitals: dict,
    risk: str,
    confidence: float,
    alerts: list,
    recommendations: list,
    ai_summary: str = "",
) -> bytes:
    """
    Build and return a PDF report as raw bytes.

    Usage in Streamlit:
        pdf_bytes = generate_pdf_report(...)
        st.download_button("Download PDF", pdf_bytes, "report.pdf", "application/pdf")
    """
    pdf = HealthReport()
    pdf.add_page()

    # Risk banner
    pdf.risk_banner(risk, confidence)

    # Worker info
    pdf.section_title("Worker Information")
    pdf.key_value("Worker ID",   worker_id)
    pdf.key_value("Report Date", datetime.now().strftime("%d %B %Y"))
    pdf.ln(3)

    # Clinical vitals
    pdf.section_title("Clinical Vitals")
    vital_labels = {
        "age":           "Age (years)",
        "bmi":           "BMI (kg/m2)",
        "glucose":       "Glucose (mg/dL)",
        "cholesterol":   "Cholesterol (mg/dL)",
        "oxygen":        "SpO2 (%)",
        "heart_rate":    "Heart Rate (bpm)",
        "heat_exposure": "Heat Exposure (/10)",
        "sleep_hours":   "Sleep Hours",
    }
    normal_ranges = {
        "glucose":     (70, 140),
        "cholesterol": (100, 200),
        "oxygen":      (95, 100),
        "heart_rate":  (60, 100),
        "bmi":         (18.5, 25),
    }

    for key, label in vital_labels.items():
        val = vitals.get(key)
        if val is None:
            continue
        lo, hi = normal_ranges.get(key, (None, None))
        flag = lo is not None and (float(val) < lo or float(val) > hi)
        display = f"{val}  [ABNORMAL]" if flag else str(val)
        pdf.key_value(label, display, highlight=flag)
    pdf.ln(3)

    # Health alerts
    if alerts:
        pdf.section_title("Health Alerts")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(180, 30, 30)
        for alert in alerts:
            clean = _safe(re.sub(r'[^\x00-\xFF]', '', str(alert)))
            pdf.cell(0, 7, f"  * {clean}", ln=True)
        pdf.ln(3)

    # Recommendations
    if recommendations:
        pdf.section_title("Recommendations")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 80, 160)
        for rec in recommendations:
            clean = _safe(re.sub(r'[^\x00-\xFF]', '', str(rec)))
            pdf.cell(0, 7, f"  * {clean}", ln=True)
        pdf.ln(3)

    # AI summary
    if ai_summary:
        pdf.section_title("AI Medical Summary")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        clean_summary = _safe(
            ai_summary
            .replace("###", "")
            .replace("##", "")
            .replace("**", "")
            .replace("*", "")
        )
        pdf.multi_cell(0, 6, clean_summary)

    return bytes(pdf.output())
