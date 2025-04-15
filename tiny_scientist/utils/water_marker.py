import os
import tempfile
import textwrap

from pypdf import PageObject, PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from rich import print


class WaterMarker:
    def _add_watermark(
        self, original_pdf_path: str, watermark_text: str, output_pdf_path: str
    ) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            watermark_pdf_path = tmp_file.name

        c = canvas.Canvas(watermark_pdf_path, pagesize=letter)
        c.saveState()
        c.translate(300, 400)
        c.rotate(45)
        c.setFillColor(Color(0.95, 0.95, 0.95))
        c.setFont("Helvetica-Bold", 28)

        max_chars_per_line = 30
        lines = textwrap.wrap(watermark_text, width=max_chars_per_line)

        line_height = 35
        y_offset = 0
        for line in lines:
            c.drawCentredString(0, y_offset, line)
            y_offset -= line_height
        c.restoreState()
        c.showPage()
        c.save()

        original_reader = PdfReader(original_pdf_path)
        watermark_reader = PdfReader(watermark_pdf_path)
        if len(watermark_reader.pages) == 0:
            print("Warning: Watermark PDF is empty. No watermark will be applied.")
            return

        watermark_page = watermark_reader.pages[0]
        writer = PdfWriter()

        for orig_page in original_reader.pages:
            new_page = PageObject.create_blank_page(
                width=orig_page.mediabox.width, height=orig_page.mediabox.height
            )

            new_page.merge_page(watermark_page)
            new_page.merge_page(orig_page)

            writer.add_page(new_page)

        with open(output_pdf_path, "wb") as out_f:
            writer.write(out_f)
        print(f"Watermarked PDF saved to: {output_pdf_path}")
        os.remove(watermark_pdf_path)
