from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ViT统一评估与同patch对照报告.pdf"


def dataframe_table(frame, widths=None):
    values = [list(frame.columns)] + frame.astype(str).values.tolist()
    table = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24557A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB5BE")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.drawRightString(landscape(A4)[0] - 14 * mm, 8 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def main():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title-cn", parent=styles["Title"], fontName="STSong-Light", fontSize=22, leading=30)
    heading = ParagraphStyle("heading-cn", parent=styles["Heading2"], fontName="STSong-Light", fontSize=14, leading=20, textColor=colors.HexColor("#174A70"))
    body = ParagraphStyle("body-cn", parent=styles["BodyText"], fontName="STSong-Light", fontSize=9.5, leading=15)
    centered = ParagraphStyle("centered-cn", parent=heading, alignment=TA_CENTER)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=14 * mm,
        title="ViT统一评估与同patch对照报告",
        author="ViT Training Project",
    )
    story = [
        Spacer(1, 18 * mm),
        Paragraph("ViT统一评估与同patch对照报告", title),
        Spacer(1, 7 * mm),
        Paragraph("生成日期：2026-09-29", body),
        Spacer(1, 10 * mm),
        Paragraph("结论适用范围", heading),
        Paragraph(
            "正式四版对比只使用共同未训练的 CRC02（6,638 张 patch），且仅有 1 例，不能据此推断总体泛化排名。"
            "其余切片可视化仅供定性查看：旧版曾训练这些病例。CRC33 两切片还存在旧基线内部的患者泄漏，"
            "当前基线严格测试需排除 CRC33。",
            body,
        ),
        Spacer(1, 5 * mm),
        Paragraph("统一评估口径", heading),
        Paragraph(
            "相同 16 通道、GT&gt;0 标签、256 像素网格、背景/未知忽略、逐通道聚合；阈值均仅在模型自身未训练过的"
            "验证集选择，验证病例不同。旧版强度不是概率；ROC/AP 为 256-bin 近似。PSNR/SSIM 不参与比较。",
            body,
        ),
        PageBreak(),
    ]

    summary = pd.read_csv(ROOT / "common_test_summary.csv").round(5)
    story += [Paragraph("共同测试集：统一多标签分类指标", heading), Spacer(1, 3 * mm), dataframe_table(summary), Spacer(1, 8 * mm)]
    baseline = pd.read_csv(ROOT / "baseline_test_scopes.csv").round(5)
    story += [Paragraph("当前基线：完整测试与去泄漏测试", heading), Spacer(1, 3 * mm), dataframe_table(baseline), Spacer(1, 7 * mm)]
    story += [
        Paragraph("颜色与 argmax 定义", heading),
        Paragraph(
            "每个通道固定一种颜色；每像素只显示最高分通道。GT 使用训练集 q 归一化后的强度 argmax；"
            "v1/v2/v3 使用还原强度并除以同一 q；当前模型使用独立 sigmoid 概率 argmax，不加 0.5 筛选。"
            "颜色图是有损展示，分类指标仍使用完整多标签。预测只按 H&amp;E 组织掩膜排除玻片背景，不利用 GT 裁切；"
            "全通道 GT 为零的像素不参与指标。",
            body,
        ),
    ]

    previews = pd.read_csv(ROOT / "preview_manifest.csv")
    max_width = landscape(A4)[0] - 28 * mm
    # Reserve vertical space for the page heading and Platypus frame paddings.
    max_height = landscape(A4)[1] - 47 * mm
    for row in previews.itertuples():
        story.append(PageBreak())
        story.append(Paragraph(f"{row.orion_slide_id} / patch {row.patch_id}", centered))
        story.append(Spacer(1, 2 * mm))
        image = Image(str(ROOT / "figures" / f"patch_{row.patch_id}_four_versions.png"))
        scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        story.append(image)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
