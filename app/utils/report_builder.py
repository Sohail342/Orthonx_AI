"""Build professional PDF diagnosis reports using ReportLab."""

import io
import os
from datetime import datetime

import httpx
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Brand Palette
BRAND_PRIMARY = colors.HexColor("#00B4D8")
BRAND_PRIMARY_DARK = colors.HexColor("#0098B0")
BRAND_DARK = colors.HexColor("#15173D")
BRAND_GRAY = colors.HexColor("#596080")
BRAND_LIGHT_GRAY = colors.HexColor("#9CA3AF")
BRAND_SURFACE = colors.HexColor("#F8FAFC")
BRAND_LIGHT_BG = colors.HexColor("#F0FDFA")
BRAND_BORDER = colors.HexColor("#E2E8F0")
BRAND_SUCCESS = colors.HexColor("#059669")
BRAND_WARNING = colors.HexColor("#D97706")
BRAND_DANGER = colors.HexColor("#DC2626")
WHITE = colors.white

LOGO_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "orthonx_logo.png")

PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 4 * cm  # 2cm margins each side


def _download_image_bytes(url: str) -> bytes | None:
    """Download an image from a URL and return its bytes."""
    if not url:
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None


def _get_styles() -> dict:
    """Create custom paragraph styles for the report."""
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=24,
            textColor=BRAND_DARK,
            spaceAfter=2 * mm,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
            leading=30,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=BRAND_GRAY,
            spaceAfter=6 * mm,
            fontName="Helvetica",
            leading=14,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=14,
            textColor=BRAND_DARK,
            spaceBefore=10 * mm,
            spaceAfter=5 * mm,
            fontName="Helvetica-Bold",
            leading=18,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=10,
            textColor=BRAND_DARK,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=3 * mm,
            fontName="Helvetica",
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            parent=base["Normal"],
            fontSize=9,
            textColor=BRAND_GRAY,
            leading=13,
            fontName="Helvetica",
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base["Normal"],
            fontSize=8,
            textColor=BRAND_LIGHT_GRAY,
            fontName="Helvetica-Bold",
            leading=12,
            spaceAfter=1 * mm,
        ),
        "value": ParagraphStyle(
            "Value",
            parent=base["Normal"],
            fontSize=10,
            textColor=BRAND_DARK,
            fontName="Helvetica",
            leading=14,
        ),
        "value_bold": ParagraphStyle(
            "ValueBold",
            parent=base["Normal"],
            fontSize=10,
            textColor=BRAND_DARK,
            fontName="Helvetica-Bold",
            leading=14,
        ),
        "tag": ParagraphStyle(
            "Tag",
            parent=base["Normal"],
            fontSize=8,
            textColor=BRAND_LIGHT_GRAY,
            fontName="Helvetica-Oblique",
            leading=11,
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=BRAND_LIGHT_GRAY,
            leading=11,
            alignment=TA_CENTER,
            fontName="Helvetica-Oblique",
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=BRAND_LIGHT_GRAY,
            alignment=TA_CENTER,
            fontName="Helvetica",
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=9,
            textColor=BRAND_GRAY,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "right_aligned": ParagraphStyle(
            "RightAligned",
            alignment=TA_RIGHT,
            leading=14,
            fontName="Helvetica",
            fontSize=10,
        ),
    }


def _build_header(styles: dict, record_data: dict) -> list:
    """Build the branded report header with logo and metadata."""
    elements = []

    # Logo
    logo_cell = ""
    if os.path.exists(LOGO_PATH):
        try:
            # Precisely fit logo height to text line height balance
            logo_cell = Image(
                LOGO_PATH, width=4.5 * cm, height=1.4 * cm, kind="proportional"
            )
            logo_cell.hAlign = "LEFT"
        except Exception:
            logo_cell = Paragraph(
                '<font size="20" color="#00B4D8"><b>Orthonx</b></font>',
                styles["title"],
            )
    else:
        logo_cell = Paragraph(
            '<font size="20" color="#00B4D8"><b>Orthonx</b></font>',
            styles["title"],
        )

    report_date = datetime.now().strftime("%B %d, %Y  •  %I:%M %p")
    pub_id = str(record_data.get("public_id", "N/A"))[:40]

    right_info = Paragraph(
        f'<font size="7" color="#9CA3AF"><b>REPORT GENERATED</b></font><br/>'
        f'<font size="10" color="#15173D"><b>{report_date}</b></font><br/>'
        f'<font size="7" color="#9CA3AF">REFERENCE: {pub_id}</font>',
        styles["right_aligned"],
    )

    header_table = Table(
        [[logo_cell, right_info]],
        colWidths=[10.5 * cm, 6.5 * cm],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    elements.append(header_table)
    elements.append(Spacer(1, 4 * mm))

    # Primary color accent bar
    elements.append(
        HRFlowable(width="100%", thickness=2.5, color=BRAND_PRIMARY, spaceAfter=1 * mm)
    )
    elements.append(
        HRFlowable(width="100%", thickness=0.5, color=BRAND_BORDER, spaceAfter=8 * mm)
    )

    return elements


def _build_title_section(styles: dict) -> list:
    """Build the report title and tagline."""
    return [
        Paragraph(
            '<font color="#00B4D8">AI</font> Diagnosis Report',
            styles["title"],
        ),
        Paragraph(
            "Automated bone fracture analysis powered by Orthonx deep learning models",
            styles["subtitle"],
        ),
    ]


def _build_patient_info(styles: dict, record_data: dict, user_data: dict) -> list:
    """Build the patient information card."""
    elements = []

    # Modern section heading
    elements.append(
        Paragraph(
            "<b>PATIENT INFORMATION</b>",
            ParagraphStyle(
                "InfoHeading",
                fontSize=9,
                textColor=BRAND_PRIMARY,
                spaceAfter=3 * mm,
                fontName="Helvetica-Bold",
                letterSpacing=1,
            ),
        )
    )

    # Parse scan date
    scan_date = "N/A"
    if record_data.get("timestamp"):
        try:
            ts = record_data["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            scan_date = ts.strftime("%B %d, %Y at %I:%M %p")
        except Exception:
            scan_date = str(record_data["timestamp"])

    # Two-column info card with clean borders
    info_data = [
        [
            Paragraph(
                "<font color='#9CA3AF' size='7.5'><b>PATIENT FULL NAME</b></font>",
                styles["label"],
            ),
            Paragraph(
                "<font color='#9CA3AF' size='7.5'><b>EMAIL ADDRESS</b></font>",
                styles["label"],
            ),
        ],
        [
            Paragraph(f"<b>{user_data.get('name', 'N/A')}</b>", styles["value_bold"]),
            Paragraph(user_data.get("email", "N/A"), styles["value"]),
        ],
        [
            Paragraph(
                "<font color='#9CA3AF' size='7.5'><b>SCAN / ANALYSIS DATE</b></font>",
                styles["label"],
            ),
            Paragraph(
                "<font color='#9CA3AF' size='7.5'><b>SYSTEM RECORD ID</b></font>",
                styles["label"],
            ),
        ],
        [
            Paragraph(scan_date, styles["value"]),
            Paragraph(
                f"<code>#{str(record_data.get('id', 'N/A'))}</code>", styles["value"]
            ),
        ],
    ]

    info_table = Table(info_data, colWidths=[CONTENT_WIDTH / 2, CONTENT_WIDTH / 2])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.5, BRAND_BORDER),
                ("LINEBELOW", (0, 1), (-1, 1), 0.1, BRAND_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 15),
                ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.append(info_table)
    elements.append(Spacer(1, 8 * mm))
    return elements


def _build_ai_summary(styles: dict, ai_summary: str) -> list:
    """Build the AI clinical summary section."""
    elements = []

    elements.append(
        Paragraph(
            "<b>AI CLINICAL SUMMARY</b>",
            ParagraphStyle(
                "InfoHeading",
                fontSize=9,
                textColor=BRAND_PRIMARY,
                spaceAfter=1 * mm,
                fontName="Helvetica-Bold",
                letterSpacing=1,
            ),
        )
    )

    elements.append(
        Paragraph(
            "Powered by Llama 3.2 Medical AI",
            styles["tag"],
        )
    )

    # Summary in a bordered card with left accent
    summary_content = Table(
        [[Paragraph(ai_summary, styles["body"])]],
        colWidths=[CONTENT_WIDTH],
    )
    summary_content.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.5, BRAND_BORDER),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, BRAND_PRIMARY),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ]
        )
    )
    elements.append(summary_content)
    elements.append(Spacer(1, 8 * mm))
    return elements


def _build_detection_table(styles: dict, record_data: dict) -> list:
    """Build the detection results table."""
    elements = []
    detections = record_data.get("diagnosis_data", {}).get("detections", [])

    elements.append(
        Paragraph(
            "<b>DETECTION RESULTS & FINDINGS</b>",
            ParagraphStyle(
                "InfoHeading",
                fontSize=9,
                textColor=BRAND_PRIMARY,
                spaceAfter=4 * mm,
                fontName="Helvetica-Bold",
                letterSpacing=1,
            ),
        )
    )

    if not detections:
        no_result_table = Table(
            [
                [
                    Paragraph(
                        '<font color="#059669"><b>✓  No fractures or abnormalities detected</b></font><br/>'
                        '<font size="9" color="#596080">The AI analysis indicates a normal radiological appearance.</font>',
                        ParagraphStyle(
                            "noResult",
                            alignment=TA_LEFT,
                            leading=16,
                            fontName="Helvetica",
                        ),
                    )
                ]
            ],
            colWidths=[CONTENT_WIDTH],
        )
        no_result_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECFDF5")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#A7F3D0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                    ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ]
            )
        )
        elements.append(no_result_table)
        return elements

    # Summary badge
    elements.append(
        Paragraph(
            f'<font size="9" color="#596080">Found <b>{len(detections)}</b> detection(s) in the analyzed image</font>',
            styles["body_small"],
        )
    )
    elements.append(Spacer(1, 2 * mm))

    # Table header
    header_style = ParagraphStyle(
        "TableHeader",
        fontSize=9,
        textColor=WHITE,
        fontName="Helvetica-Bold",
        leading=12,
    )
    table_header = [
        Paragraph("#", header_style),
        Paragraph("Finding", header_style),
        Paragraph("Confidence", header_style),
        Paragraph("Bounding Box", header_style),
    ]
    table_rows = [table_header]

    for i, det in enumerate(detections):
        box = det.get("box", {})
        loc = f"({box.get('x1', 0)}, {box.get('y1', 0)}) → ({box.get('x2', 0)}, {box.get('y2', 0)})"
        conf = det.get("confidence", 0)
        conf_pct = f"{conf * 100:.0f}%"

        # Color-code confidence
        if conf >= 0.7:
            conf_color = "#059669"
            conf_label = "HIGH"
        elif conf >= 0.4:
            conf_color = "#D97706"
            conf_label = "MED"
        else:
            conf_color = "#DC2626"
            conf_label = "LOW"

        table_rows.append(
            [
                Paragraph(f"<b>{i + 1}</b>", styles["value"]),
                Paragraph(
                    f"<b>{det.get('class', 'Unknown').title()}</b>",
                    styles["value_bold"],
                ),
                Paragraph(
                    f'<font color="{conf_color}"><b>{conf_pct}</b></font> '
                    f'<font size="7" color="{conf_color}">({conf_label})</font>',
                    styles["value"],
                ),
                Paragraph(
                    f'<font size="8" color="#596080">{loc}</font>', styles["value"]
                ),
            ]
        )

    det_table = Table(
        table_rows,
        colWidths=[1.2 * cm, 5 * cm, 3.5 * cm, 7.3 * cm],
    )
    det_table.setStyle(
        TableStyle(
            [
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                # Data rows
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_SURFACE]),
                # Grid
                ("INNERGRID", (0, 0), (-1, -1), 0.25, BRAND_BORDER),
                ("BOX", (0, 0), (-1, -1), 0.75, BRAND_BORDER),
                # Alignment
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # Padding
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    elements.append(det_table)
    return elements


def _build_images_section(styles: dict, record_data: dict) -> list:
    """Build the diagnostic images section with embedded images."""
    elements = []

    elements.append(
        Paragraph(
            "<b>DIAGNOSTIC IMAGES & HEATMAPS</b>",
            ParagraphStyle(
                "InfoHeading",
                fontSize=9,
                textColor=BRAND_PRIMARY,
                spaceAfter=5 * mm,
                fontName="Helvetica-Bold",
                letterSpacing=1,
            ),
        )
    )

    image_entries = [
        ("Original X-ray", record_data.get("uploaded_image_url")),
        ("AI Detection Result", record_data.get("result_image_url")),
        ("Grad-CAM Heatmap", record_data.get("gradcam_image_url")),
    ]

    for label, url in image_entries:
        if not url:
            continue
        img_bytes = _download_image_bytes(url)
        if not img_bytes:
            continue

        try:
            img_io = io.BytesIO(img_bytes)
            img = Image(img_io, width=13 * cm, height=13 * cm, kind="proportional")
            img.hAlign = "CENTER"

            caption = Paragraph(label, styles["caption"])

            img_table = Table(
                [[caption], [img]],
                colWidths=[CONTENT_WIDTH],
            )
            img_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BOX", (0, 0), (-1, -1), 0.5, BRAND_BORDER),
                        ("BACKGROUND", (0, 0), (-1, 0), BRAND_SURFACE),
                        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
                        ("TOPPADDING", (0, 0), (0, 0), 8),
                        ("BOTTOMPADDING", (0, -1), (0, -1), 10),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            elements.append(img_table)
            elements.append(Spacer(1, 6 * mm))
        except Exception as e:
            logger.warning(f"Failed to embed image '{label}': {e}")

    return elements


def _build_footer(styles: dict) -> list:
    """Build the disclaimer and footer."""
    elements = []

    elements.append(Spacer(1, 12 * mm))
    elements.append(
        HRFlowable(width="100%", thickness=0.5, color=BRAND_BORDER, spaceAfter=5 * mm)
    )

    # Disclaimer box
    disclaimer_table = Table(
        [
            [
                Paragraph(
                    '<font color="#DC2626"><b>⚠ DISCLAIMER</b></font><br/><br/>'
                    "This report has been generated by an AI-assisted diagnostic system "
                    "and is intended for informational purposes only. It should <b>NOT</b> be used as a "
                    "substitute for professional medical advice, diagnosis, or treatment. Always seek "
                    "the guidance of a qualified healthcare provider with any questions regarding a "
                    "medical condition. The AI model may produce inaccurate results and all findings "
                    "should be verified by a certified radiologist.",
                    styles["disclaimer"],
                )
            ]
        ],
        colWidths=[CONTENT_WIDTH],
    )
    disclaimer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#FECACA")),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    elements.append(disclaimer_table)

    elements.append(Spacer(1, 6 * mm))
    elements.append(
        Paragraph(
            f"© {datetime.now().year} <b>Orthonx</b>  —  AI-Powered Bone Fracture Detection Platform",
            styles["footer"],
        )
    )
    elements.append(
        Paragraph(
            "This document is confidential and intended solely for the named recipient.",
            styles["footer"],
        )
    )

    return elements


def build_diagnosis_pdf(
    record_data: dict,
    user_data: dict,
    ai_summary: str,
) -> bytes:
    """Generate a professional PDF diagnosis report.

    Args:
        record_data: Dict with keys: id, public_id, timestamp, diagnosis_data,
                     uploaded_image_url, result_image_url, gradcam_image_url
        user_data:   Dict with keys: name, email
        ai_summary:  AI-generated clinical narrative string.

    Returns:
        PDF file contents as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2 * cm,
        title="Orthonx AI Diagnosis Report",
        author="Orthonx AI Platform",
    )

    styles = _get_styles()
    elements = []

    # Page 1: Header, Patient Info, AI Summary, Detection Table
    elements.extend(_build_header(styles, record_data))
    elements.extend(_build_title_section(styles))
    elements.extend(_build_patient_info(styles, record_data, user_data))
    elements.extend(_build_ai_summary(styles, ai_summary))
    elements.extend(_build_detection_table(styles, record_data))

    # Page 2: Images
    image_elements = _build_images_section(styles, record_data)
    if image_elements:
        elements.append(PageBreak())
        elements.extend(_build_header(styles, record_data))
        elements.extend(image_elements)

    # Footer / Disclaimer
    elements.extend(_build_footer(styles))

    # Build
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"PDF report built successfully ({len(pdf_bytes)} bytes)")
    return pdf_bytes
