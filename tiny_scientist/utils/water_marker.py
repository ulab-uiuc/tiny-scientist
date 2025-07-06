import textwrap
from io import BytesIO

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


class WaterMarker:
    def __init__(self, opacity: float = 0.1, font_size: int = 36):
        self.opacity = opacity
        self.font_size = font_size

    def _add_watermark(
        self, original_pdf_path: str, watermark_text: str, output_pdf_path: str
    ) -> None:
        reader = PdfReader(original_pdf_path)
        writer = PdfWriter()

        for page in reader.pages:
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)

            # Create watermark canvas in memory
            packet = BytesIO()
            c = canvas.Canvas(packet, pagesize=(page_width, page_height))

            # Configure font size relative to page width
            base_font_size = (
                min(page_width, page_height) * 0.035
            )  # ~3.5% of min dimension
            c.setFont("Helvetica-Bold", base_font_size)
            c.setFillColor(
                Color(0.6, 0.6, 0.6, alpha=0.06)
            )  # very low opacity to avoid disruption

            # Wrap long watermark text to fit the diagonal
            max_chars = 25
            lines = textwrap.wrap(watermark_text, width=max_chars)

            c.saveState()
            c.translate(page_width / 2, page_height / 2)
            c.rotate(45)

            spacing = base_font_size + 8
            total_height = spacing * len(lines)
            y_start = total_height / 2

            for i, line in enumerate(lines):
                y = y_start - i * spacing
                c.drawCentredString(0, y, line)

            c.restoreState()
            c.save()
            packet.seek(0)

            # Merge watermark with the page
            watermark_pdf = PdfReader(packet)
            watermark_page = watermark_pdf.pages[0]
            page.merge_page(watermark_page)
            writer.add_page(page)

        # Output the watermarked PDF
        with open(output_pdf_path, "wb") as f:
            writer.write(f)
