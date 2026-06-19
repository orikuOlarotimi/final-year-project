"""
test_image_captions.py
----------------------
Test script: screenshots every page in a PDF that contains at least one image
(one screenshot per page, no duplicates), sends the full page to GPT-4.1 mini,
and asks it to identify and describe every figure on that page using both the
visual content and the surrounding text/labels it can see.

No embedding, no Pinecone — just raw captions printed to terminal so you can
verify quality before committing to the main pipeline.

Usage:
    python test_image_captions.py path/to/your.pdf
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_IMAGE_WIDTH  = 50    # px — skip tiny decorative images when deciding
MIN_IMAGE_HEIGHT = 50    # px   whether a page qualifies as an "image page"
PAGE_RENDER_DPI  = 150   # DPI for full-page screenshot (150 = clear, not huge)

client = AsyncOpenAI()   # reads OPENAI_API_KEY from environment

# ---------------------------------------------------------------------------
# Step 1 — find pages that contain images, screenshot each once
# ---------------------------------------------------------------------------


def find_image_pages(pdf_path: str) -> list[dict]:
    """
    Scan every page. Screenshots a page if it contains either:
      (a) at least one embedded image object above the size threshold, OR
      (b) significant vector drawing content (>20 paths) — catches flowcharts
          and diagrams drawn directly in the PDF with no embedded image object.

    One screenshot per page, no duplicates.

    Returns a list of dicts:
        page_num   : 1-indexed page number
        png_bytes  : full-page PNG screenshot
        image_count: qualifying embedded images found (0 if vector-only page)
        reason     : "embedded_image" | "vector_drawing" | "both"
    """
    results = []
    doc     = fitz.open(pdf_path)
    scale   = PAGE_RENDER_DPI / 72
    mat     = fitz.Matrix(scale, scale)

    for page_num, page in enumerate(doc):

        # --- Check 1: embedded image objects --------------------------------
        qualifying = []
        for img_info in page.get_images(full=True):
            xref       = img_info[0]
            base_image = doc.extract_image(xref)
            w          = base_image.get("width", 0)
            h          = base_image.get("height", 0)
            if w >= MIN_IMAGE_WIDTH and h >= MIN_IMAGE_HEIGHT:
                qualifying.append(img_info)

        has_images = len(qualifying) > 0

        # --- Check 2: vector drawings (flowcharts, diagrams, etc.) ----------
        # Almost every page has a few paths (borders, underlines).
        # A real diagram typically has >20 distinct drawing paths.
        drawing_count = len(page.get_drawings())
        has_drawings  = drawing_count > 20

        if not has_images and not has_drawings:
            continue   # plain text page — skip

        # Screenshot the full page once
        pix       = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")

        if has_images and has_drawings:
            reason = "both"
        elif has_images:
            reason = "embedded_image"
        else:
            reason = "vector_drawing"

        results.append({
            "page_num":    page_num + 1,
            "png_bytes":   png_bytes,
            "image_count": len(qualifying),
            "reason":      reason,
        })

        print(
            f"  Page {page_num + 1}: screenshotted "
            f"[{reason}] — {len(qualifying)} embedded image(s), "
            f"{drawing_count} drawing path(s)."
        )

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Step 2 — GPT-4.1 mini analyses the full page and returns per-figure JSON
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are a document analysis assistant. You will be given a screenshot of a full
page from an academic or technical PDF document.

Your job is to identify every distinct figure, diagram, chart, or image on the
page and produce a structured description of each one.

For each figure you find, return a JSON object with these fields:
  - "label"       : the figure label exactly as written (e.g. "Fig. 4", "Figure 2.1").
                    If no label is visible, use "Unlabelled figure".
  - "description" : a detailed, self-contained description of the figure. Include:
                      * Every visible label, annotation, step number, and text element
                      * The type of diagram (flowchart, bar chart, architecture diagram, etc.)
                      * All arrows and what they connect
                      * All decision branches and their outcomes (Yes/No, True/False, etc.)
                      * Any data values, axis labels, or legends
                      * The overall meaning or process the figure communicates
                    Write as if describing to someone who cannot see the image.

Return ONLY a JSON array of these objects — no preamble, no markdown fences.
Example format:
[
  {"label": "Fig. 1", "description": "..."},
  {"label": "Fig. 2", "description": "..."}
]

If you find no figures at all on the page, return an empty array: []
"""


async def analyse_page(page: dict) -> list[dict]:
    """
    Send a full-page screenshot to GPT-4.1 mini.
    Returns a list of {label, description} dicts — one per figure found.
    """
    b64 = base64.standard_b64encode(page["png_bytes"]).decode("utf-8")

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is page {page['page_num']} of the document. "
                            "Please identify and describe every figure on this page."
                        ),
                    },
                ],
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        )

    try:
        figures = json.loads(raw)
        if not isinstance(figures, list):
            figures = []
    except json.JSONDecodeError:
        # If JSON parsing fails, wrap the raw text as a single unlabelled entry
        figures = [{"label": "Unlabelled figure", "description": raw}]

    return figures


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run(pdf_path: str):
    print(f"\n{'='*60}")
    print(f"PDF : {pdf_path}")
    print(f"{'='*60}\n")

    print("Scanning for image pages...\n")
    pages = find_image_pages(pdf_path)

    if not pages:
        print("\nNo image-containing pages found in this PDF.")
        return

    # ── Save page screenshots so you can inspect them ───────────────────────
    pdf_stem   = Path(pdf_path).stem
    output_dir = Path("extracted_images") / pdf_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for page in pages:
        filename = f"page{page['page_num']:03d}_screenshot.png"
        (output_dir / filename).write_bytes(page["png_bytes"])

    print(f"\nSaved {len(pages)} page screenshot(s) to: {output_dir.resolve()}\n")
    # ────────────────────────────────────────────────────────────────────────

    print(f"Sending {len(pages)} page(s) to GPT-4.1 mini for analysis...\n")

    # Analyse all pages concurrently
    tasks   = [analyse_page(p) for p in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_figures = 0

    for page, result in zip(pages, results):
        print(f"{'='*60}")
        print(f"PAGE {page['page_num']}  ({page['image_count']} image object(s) detected)")
        print(f"Screenshot: page{page['page_num']:03d}_screenshot.png")
        print(f"{'='*60}\n")

        if isinstance(result, Exception):
            print(f"  ERROR analysing page: {result}\n")
            continue

        if not result:
            print("  No figures identified on this page.\n")
            continue

        for i, fig in enumerate(result, 1):
            total_figures += 1
            print(f"  Figure {i} — {fig.get('label', 'Unlabelled')}")
            print(f"  {'─'*54}")
            print(f"  {fig.get('description', '(no description)')}")
            print()

    print(f"{'='*60}")
    print(f"Done.  Pages screenshotted: {len(pages)}  |  Figures identified: {total_figures}")
    print(f"Screenshots saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image_captions.py path/to/your.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    asyncio.run(run(pdf_path))