from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import CLASS_LABELS, DISCLAIMER, POSITIVE_CLASS

def get_medical_interpretation(final_class: int, modality: str = "image") -> str:
    if final_class == POSITIVE_CLASS:
        if modality == "video":
            return (
                "The endoscopic video analysis indicates clear signs of gastrointestinal inflammation across multiple segments. "
                "Visual features and automated neural heatmaps are consistent with active inflammatory mucosal changes."
            )
        return (
            "The endoscopic image indicates signs of gastrointestinal inflammation. "
            "Visual features are consistent with inflammatory mucosal changes."
        )
    
    if modality == "video":
        return "No significant inflammatory findings were detected across the examined endoscopic video streams."
    return "No significant inflammatory findings were detected in the examined endoscopic image."


def get_clinical_recommendation(final_class: int, modality: str = "image") -> str:
    if final_class == POSITIVE_CLASS:
        return (
            "Clinical correlation is recommended. Please refer the patient to a gastroenterology "
            "specialist for confirmatory assessment, histopathological evaluation (biopsy) if indicated, "
            "and appropriate targeted treatment planning."
        )
    
    if modality == "video":
        return (
            "Based on the AI-assisted review, no urgent inflammatory pattern was detected. "
            "Routine follow-up with a gastroenterology specialist is advised if clinical symptoms persist."
        )
    return (
        "Based on the AI-assisted review, no urgent inflammatory pattern was detected. "
        "Routine follow-up with a gastroenterology specialist is advised if symptoms persist."
    )


def _temp_image_path(image_rgb, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    Image.fromarray(image_rgb).save(path)
    return path



def generate_image_pdf_report(
    result,
    patient_name: str,
    patient_phone: str,
    image_name: str,
    output_path: Path,
    examination_time: datetime | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "report_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    exam_time = examination_time or datetime.now()
    interpretation = get_medical_interpretation(result.predicted_class, modality="image")
    recommendation = get_clinical_recommendation(result.predicted_class, modality="image")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], textColor=colors.HexColor("#0B4F6C"), spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=colors.HexColor("#145C9E"), spaceAfter=8)
    body_style = styles["BodyText"]
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["BodyText"], textColor=colors.HexColor("#666666"), fontSize=9)

    story = [
        Paragraph("Endoscopy Diagnostic System — Image Analysis Report", title_style),
        Paragraph(f"Report Generated: {exam_time.strftime('%Y-%m-%d %H:%M:%S')}", body_style),
        Spacer(1, 0.4 * cm),
        Paragraph("Patient Information", heading_style),
    ]

    patient_table = Table(
        [
            ["Patient Name", patient_name or "N/A"],
            ["Phone Number", patient_phone or "N/A"],
            ["Date and Time of Examination", exam_time.strftime("%Y-%m-%d %H:%M:%S")],
        ],
        colWidths=[5.5 * cm, 10.5 * cm],
    )
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F4F8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([patient_table, Spacer(1, 0.5 * cm), Paragraph("Examination Results", heading_style)])

    results_table = Table(
        [
            ["Uploaded Image", image_name],
            ["Classification Result", result.label],
            ["Confidence Score", f"{result.probability * 100:.1f}%"],
        ],
        colWidths=[5.5 * cm, 10.5 * cm],
    )
    verdict_color = colors.HexColor("#B42318") if result.predicted_class == POSITIVE_CLASS else colors.HexColor("#027A48")
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (1, 1), (1, 1), verdict_color),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))

    story.extend([
        results_table,
        Spacer(1, 0.5 * cm),
        Paragraph("Medical Interpretation", heading_style),
        Paragraph(interpretation, body_style),
        Spacer(1, 0.4 * cm),
        Paragraph("Recommendations", heading_style),
        Paragraph(recommendation, body_style),
        Spacer(1, 0.5 * cm),
        Paragraph("Analyzed Region of Interest & Model Activation (Grad-CAM)", heading_style),
        Spacer(1, 0.2 * cm),
    ])

    roi_path = _temp_image_path(result.roi_rgb, temp_dir, "analyzed_roi.png")
    heatmap_path = _temp_image_path(result.heatmap_rgb, temp_dir, "heatmap_roi.png")

    image_table = Table(
        [
            [
                RLImage(str(roi_path), width=7.5 * cm, height=7.5 * cm), 
                RLImage(str(heatmap_path), width=7.5 * cm, height=7.5 * cm)
            ],
            [
                Paragraph("Original Preprocessed ROI", ParagraphStyle("centered", alignment=1, fontSize=9)), 
                Paragraph("Model Focus (Heatmap)", ParagraphStyle("centered", alignment=1, fontSize=9))
            ]
        ],
        colWidths=[8.5 * cm, 8.5 * cm],
    )
    
    story.append(image_table)
    story.append(Spacer(1, 0.5 * cm))
    story.extend([Paragraph("Disclaimer", heading_style), Paragraph(DISCLAIMER, disclaimer_style)])
    
    doc.build(story)
    return output_path



def generate_video_pdf_report(
    result,
    patient_name: str,
    patient_phone: str,
    video_name: str,
    output_path: Path,
    examination_time: datetime | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "report_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    exam_time = examination_time or datetime.now()
    interpretation = get_medical_interpretation(result.final_class, modality="video")
    recommendation = get_clinical_recommendation(result.final_class, modality="video")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], textColor=colors.HexColor("#0B4F6C"), spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=colors.HexColor("#145C9E"), spaceAfter=8)
    body_style = styles["BodyText"]
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["BodyText"], textColor=colors.HexColor("#666666"), fontSize=9)

    story = [
        Paragraph("Endoscopy Diagnostic System — Video Analysis Report", title_style),
        Paragraph(f"Report Generated: {exam_time.strftime('%Y-%m-%d %H:%M:%S')}", body_style),
        Spacer(1, 0.4 * cm),
        Paragraph("Patient Information", heading_style),
    ]

    patient_table = Table(
        [
            ["Patient Name", patient_name or "N/A"],
            ["Phone Number", patient_phone or "N/A"],
            ["Date and Time of Examination", exam_time.strftime("%Y-%m-%d %H:%M:%S")],
        ],
        colWidths=[5.5 * cm, 10.5 * cm],
    )
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F4F8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([patient_table, Spacer(1, 0.5 * cm), Paragraph("Examination Results Summary", heading_style)])

    results_table = Table(
        [
            ["Uploaded Video File", video_name],
            ["Overall Classification", result.final_label],
            ["AI Confidence Score", f"{result.confidence * 100:.1f}%"],
            ["Total Frames Processed", str(result.total_frames)],
            ["Inflammation Frames Found", str(result.positive_frames)],
        ],
        colWidths=[5.5 * cm, 10.5 * cm],
    )
    verdict_color = colors.HexColor("#B42318") if result.final_class == POSITIVE_CLASS else colors.HexColor("#027A48")
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (1, 1), (1, 1), verdict_color),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))

    story.extend([
        results_table,
        Spacer(1, 0.5 * cm),
        Paragraph("Medical Interpretation", heading_style),
        Paragraph(interpretation, body_style),
        Spacer(1, 0.4 * cm),
        Paragraph("Recommendations", heading_style),
        Paragraph(recommendation, body_style),
        Spacer(1, 0.6 * cm),
        Paragraph("🧠 Targeted Neural Activation Findings (Grad-CAM Per Frame)", heading_style),
        Paragraph("Below are the key frame sequences highlighting specific regions where the AI model localized inflammation patterns:", body_style),
        Spacer(1, 0.3 * cm),
    ])

    target_frames = result.key_frames if hasattr(result, "key_frames") else result.frame_predictions

    if target_frames and len(target_frames) > 0:
        for idx, frame in enumerate(target_frames):
            if hasattr(frame, "roi_rgb") and frame.roi_rgb is not None:
                roi_path = _temp_image_path(frame.roi_rgb, temp_dir, f"roi_frame_{idx}.png")
                heatmap_matrix = getattr(frame, "heatmap_rgb", frame.roi_rgb)
                heatmap_path = _temp_image_path(heatmap_matrix, temp_dir, f"heatmap_frame_{idx}.png")

                timestamp_text = f"Frame #{frame.frame_index} — Timestamp: {frame.timestamp_sec:.1f}s | Confidence: {frame.probability*100:.1f}% ({frame.label})"
                story.append(Paragraph(timestamp_text, ParagraphStyle(f"FrameTitle_{idx}", parent=styles["Normal"], fontName="Helvetica-Bold", textColor=colors.HexColor("#1A202C"), spaceAfter=4)))

                image_table = Table(
                    [
                        [
                            RLImage(str(roi_path), width=7.2 * cm, height=7.2 * cm), 
                            RLImage(str(heatmap_path), width=7.2 * cm, height=7.2 * cm)
                        ],
                        [
                            Paragraph("Original Preprocessed ROI", ParagraphStyle(f"c1_{idx}", alignment=1, fontSize=8, textColor=colors.gray)), 
                            Paragraph("Model Focus Attention (Heatmap)", ParagraphStyle(f"c2_{idx}", alignment=1, fontSize=8, textColor=colors.HexColor("#B42318")))
                        ]
                    ],
                    colWidths=[8.2 * cm, 8.2 * cm],
                )
                image_table.setStyle(TableStyle([
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
                ]))
                
                story.append(image_table)
                story.append(Spacer(1, 0.4 * cm))
    else:
        story.append(Paragraph("No representative abnormal frames were extracted for visual visualization.", body_style))
        story.append(Spacer(1, 0.5 * cm))

    story.extend([Spacer(1, 0.3 * cm), Paragraph("Regulatory Disclaimer", heading_style), Paragraph(DISCLAIMER, disclaimer_style)])
    
    doc.build(story)
    return output_path