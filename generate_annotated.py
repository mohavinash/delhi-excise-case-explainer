#!/usr/bin/env python3
"""
Generate an annotated version of the Delhi Excise Policy explainer blog.

Automatically:
1. Parses index.html for all court-quote blocks + paragraph references
2. Maps paragraph numbers to PDF pages
3. Finds and highlights quote text in the PDF using smart multi-fragment search
4. Renders cropped, highlighted screenshots
5. Produces index-annotated.html with embedded PDF screenshots

Usage: python3 generate_annotated.py
"""

import fitz  # PyMuPDF
import re
import os
import html
from html.parser import HTMLParser
from PIL import Image
import io


# === CONFIG ===
PDF_PATH = "/Users/avinash/Downloads/CBI_v_Kuldeep_Singh___Ors.pdf"
HTML_PATH = "index.html"
OUTPUT_HTML = "index-annotated.html"
IMG_DIR = "pdf-highlights"
RENDER_SCALE = 2.5        # 2.5x for crisp retina-quality text
CONTEXT_PADDING = 80      # pts above/below the highlighted region
SIDE_MARGIN = 25          # pts trimmed from left/right edges
HIGHLIGHT_COLOR = (1, 0.92, 0.23)  # warm yellow


# === STEP 1: Extract quotes from HTML ===

class QuoteExtractor(HTMLParser):
    """Extract all court-quote blocks and their paragraph references from HTML."""

    def __init__(self):
        super().__init__()
        self.quotes = []
        self._in_quote = False
        self._in_para_ref = False
        self._current_text = []
        self._current_ref = []

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "")
        if tag == "div" and "court-quote" in classes:
            self._in_quote = True
            self._current_text = []
            self._current_ref = []
        elif tag == "span" and "para-ref" in classes:
            self._in_para_ref = True

    def handle_endtag(self, tag):
        if tag == "span" and self._in_para_ref:
            self._in_para_ref = False
        elif tag == "div" and self._in_quote:
            self._in_quote = False
            text = " ".join("".join(self._current_text).split()).strip()
            ref = " ".join("".join(self._current_ref).split()).strip()
            if text and ref:
                self.quotes.append({"text": text, "ref": ref})

    def handle_data(self, data):
        if self._in_para_ref:
            self._current_ref.append(data)
        elif self._in_quote:
            self._current_text.append(data)

    def handle_entityref(self, name):
        char = html.unescape(f"&{name};")
        if self._in_para_ref:
            self._current_ref.append(char)
        elif self._in_quote:
            self._current_text.append(char)

    def handle_charref(self, name):
        char = html.unescape(f"&#{name};")
        if self._in_para_ref:
            self._current_ref.append(char)
        elif self._in_quote:
            self._current_text.append(char)


def extract_quotes(html_path):
    """Parse HTML and return list of {text, ref} dicts."""
    with open(html_path, "r") as f:
        content = f.read()
    parser = QuoteExtractor()
    parser.feed(content)
    print(f"[1/5] Extracted {len(parser.quotes)} court quotes from {html_path}")
    for i, q in enumerate(parser.quotes):
        para = q["ref"]
        preview = q["text"][:80] + "..." if len(q["text"]) > 80 else q["text"]
        print(f"  [{i+1}] {para}: {preview}")
    return parser.quotes


# === STEP 2: Build paragraph-to-page index ===

def build_para_page_index(doc):
    """Scan PDF to map paragraph numbers to page numbers."""
    index = {}
    # Pattern: paragraph number at start of line (e.g., "1077." or "357.")
    para_pattern = re.compile(r'\b(\d{1,4})\.\s')

    for page_num in range(doc.page_count):
        text = doc[page_num].get_text()
        for match in para_pattern.finditer(text):
            para_num = int(match.group(1))
            if para_num not in index:
                index[para_num] = page_num

    print(f"[2/5] Indexed {len(index)} paragraph numbers across {doc.page_count} pages")
    return index


def parse_para_numbers(ref_text):
    """Extract paragraph numbers from reference text like 'Paragraph 1077' or 'Para 507'."""
    numbers = re.findall(r'\d+', ref_text)
    return [int(n) for n in numbers]


# === STEP 3: Smart multi-fragment search and highlight ===

def generate_search_fragments(text, min_words=3, max_words=5):
    """Break quote text into overlapping fragments for robust PDF search.

    Strategy: extract multiple short, unique phrases from different parts
    of the quote. Short phrases are more likely to appear on a single PDF line.
    """
    # Clean the text
    clean = re.sub(r'\s+', ' ', text).strip()
    # Remove common punctuation that might differ in PDF
    clean = clean.replace('\u2018', "'").replace('\u2019', "'")
    clean = clean.replace('\u201c', '"').replace('\u201d', '"')

    words = clean.split()
    fragments = []

    if len(words) <= max_words:
        fragments.append(clean)
        return fragments

    # Take fragments from start, middle, and end
    # Also take every Nth fragment for coverage
    step = max(1, len(words) // 6)
    for i in range(0, len(words) - min_words + 1, step):
        frag = " ".join(words[i:i + min_words])
        # Skip very short or common fragments
        if len(frag) > 10:
            fragments.append(frag)

    # Always include first and last fragments
    fragments.insert(0, " ".join(words[:min_words]))
    fragments.append(" ".join(words[-min_words:]))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in fragments:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def normalize_word(w):
    """Normalize a word for fuzzy matching (lowercase, strip punctuation)."""
    w = w.lower().strip()
    w = re.sub(r'[.,;:!?\'"()\[\]{}\u2018\u2019\u201c\u201d\u2014\u2013\u2026]', '', w)
    return w


def get_quote_words(quote_text):
    """Normalize quote text into a list of cleaned words."""
    clean = re.sub(r'\s+', ' ', quote_text).strip()
    clean = clean.replace('\u2018', "'").replace('\u2019', "'")
    clean = clean.replace('\u201c', '"').replace('\u201d', '"')
    words = [normalize_word(w) for w in clean.split()]
    return [w for w in words if w]


def _find_sequence_start(page_words_norm, target_words, max_scan=None):
    """Find where target_words starts in page_words_norm. Returns index or None."""
    limit = len(page_words_norm)
    if max_scan is not None:
        limit = min(limit, max_scan)
    for search_len in (5, 4, 3):
        if search_len > len(target_words):
            continue
        target = target_words[:search_len]
        for i in range(limit - search_len + 1):
            if all(page_words_norm[i + j] == target[j] for j in range(search_len)):
                return i
    return None


def find_full_quote_rects(page, quote_text):
    """Find exact word-level rectangles covering the full quote on the page.

    Returns (rects, words_matched, total_quote_words).
    words_matched < total_quote_words means the quote continues onto the next page.
    """
    words_data = page.get_text("words")
    if not words_data:
        return [], 0, 0

    quote_words = get_quote_words(quote_text)
    if len(quote_words) < 3:
        return [], 0, len(quote_words)

    page_words_norm = [normalize_word(w[4]) for w in words_data]

    # Find the start of the quote
    start_idx = _find_sequence_start(page_words_norm, quote_words)
    if start_idx is None:
        return [], 0, len(quote_words)

    # Find the end of the quote using last N words
    end_idx = None
    for search_len in (5, 4, 3):
        if search_len > len(quote_words):
            continue
        target = quote_words[-search_len:]
        lo = max(start_idx, start_idx + len(quote_words) - search_len - 10)
        hi = min(len(page_words_norm) - search_len + 1, start_idx + len(quote_words) + 20)
        for i in range(lo, hi):
            if all(page_words_norm[i + j] == target[j] for j in range(search_len)):
                end_idx = i + search_len - 1
                break
        if end_idx is not None:
            break

    if end_idx is not None:
        # Full match on this page
        rects = [fitz.Rect(words_data[i][:4]) for i in range(start_idx, end_idx + 1)]
        return rects, len(quote_words), len(quote_words)

    # Partial match — quote continues beyond this page
    # Collect from start to last word on the page
    end_idx = len(words_data) - 1
    rects = [fitz.Rect(words_data[i][:4]) for i in range(start_idx, end_idx + 1)]
    words_matched = end_idx - start_idx + 1
    return rects, words_matched, len(quote_words)


def find_continuation_rects(page, quote_words, skip_n):
    """Find the remaining portion of a quote at the top of the next page."""
    words_data = page.get_text("words")
    if not words_data:
        return []

    remaining = quote_words[skip_n:]
    if len(remaining) < 3:
        return []

    page_words_norm = [normalize_word(w[4]) for w in words_data]

    # The continuation should be near the top of the page (within first ~50 words)
    start_idx = _find_sequence_start(page_words_norm, remaining, max_scan=50)
    if start_idx is None:
        return []

    # Find where remaining text ends
    end_idx = None
    for search_len in (5, 4, 3):
        if search_len > len(remaining):
            continue
        target = remaining[-search_len:]
        lo = start_idx
        hi = min(len(page_words_norm) - search_len + 1, start_idx + len(remaining) + 10)
        for i in range(lo, hi):
            if all(page_words_norm[i + j] == target[j] for j in range(search_len)):
                end_idx = i + search_len - 1
                break
        if end_idx is not None:
            break

    if end_idx is None:
        end_idx = min(start_idx + len(remaining) - 1, len(words_data) - 1)

    return [fitz.Rect(words_data[i][:4]) for i in range(start_idx, end_idx + 1)]


def merge_rects_to_lines(rects):
    """Merge individual word rects into per-line highlight rects."""
    if not rects:
        return []

    # Group by approximate y-center (words on the same line)
    lines = {}
    for r in rects:
        center_y = round((r.y0 + r.y1) / 2)
        matched_key = None
        for key in lines:
            if abs(key - center_y) < 5:
                matched_key = key
                break
        if matched_key is not None:
            lines[matched_key].append(r)
        else:
            lines[center_y] = [r]

    # For each line, create a single rect spanning all words
    merged = []
    for line_y in sorted(lines.keys()):
        lr = lines[line_y]
        merged.append(fitz.Rect(
            min(r.x0 for r in lr),
            min(r.y0 for r in lr),
            max(r.x1 for r in lr),
            max(r.y1 for r in lr),
        ))

    return merged


def find_and_highlight(doc, quote, para_page_index):
    """Find a quote in the PDF and return list of (page_num, line_rects) tuples.

    Usually returns 1 tuple. Returns 2 if the quote spans a page break.
    """
    para_nums = parse_para_numbers(quote["ref"])

    # Determine candidate pages: target para's page +/- 2 pages
    candidate_pages = set()
    for pn in para_nums:
        if pn in para_page_index:
            base = para_page_index[pn]
            for offset in range(-2, 3):
                pg = base + offset
                if 0 <= pg < doc.page_count:
                    candidate_pages.add(pg)

    # If no para mapping found, search broadly (last 100 pages for concluding paras)
    if not candidate_pages:
        for pg in range(max(0, doc.page_count - 100), doc.page_count):
            candidate_pages.add(pg)

    fragments = generate_search_fragments(quote["text"])

    # Phase 1: Find the page with the most fragment hits
    best_page = None
    best_frag_rects = []
    best_score = 0

    for page_num in sorted(candidate_pages):
        page = doc[page_num]
        page_rects = []
        hits = 0

        for frag in fragments:
            instances = page.search_for(frag)
            if instances:
                hits += 1
                page_rects.extend(instances)

        if hits > best_score:
            best_score = hits
            best_page = page_num
            best_frag_rects = page_rects

    if best_page is None or not best_frag_rects:
        return []

    # Phase 2: Get full continuous quote rects via word-level matching
    page = doc[best_page]
    word_rects, matched, total = find_full_quote_rects(page, quote["text"])

    if word_rects and matched >= total:
        # Full match on one page
        return [(best_page, merge_rects_to_lines(word_rects))]

    if word_rects and matched < total:
        # Partial match — quote spans to the next page
        result = [(best_page, merge_rects_to_lines(word_rects))]
        next_pg = best_page + 1
        if next_pg < doc.page_count:
            quote_words = get_quote_words(quote["text"])
            cont_rects = find_continuation_rects(doc[next_pg], quote_words, matched)
            if cont_rects:
                result.append((next_pg, merge_rects_to_lines(cont_rects)))
        return result

    # Fallback: fragment rects if word matching found nothing
    return [(best_page, best_frag_rects)]


def _render_one_crop(doc, page_num, rects, pad_top=CONTEXT_PADDING, pad_bottom=CONTEXT_PADDING):
    """Highlight rects on a page, render a cropped pixmap, clean up. Returns PIL Image."""
    page = doc[page_num]

    for rect in rects:
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=HIGHLIGHT_COLOR)
        annot.update()

    min_y = min(r.y0 for r in rects)
    max_y = max(r.y1 for r in rects)

    crop = fitz.Rect(
        SIDE_MARGIN,
        max(0, min_y - pad_top),
        page.rect.width - SIDE_MARGIN,
        min(page.rect.height, max_y + pad_bottom)
    )

    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE), clip=crop)

    # Clean up annotations
    for annot in list(page.annots()):
        page.delete_annot(annot)

    # Convert to PIL Image
    return Image.open(io.BytesIO(pix.tobytes("png")))


def highlight_and_render(doc, page_rects_list, filename, img_dir):
    """Highlight and render one or more page crops, stitching if needed.

    page_rects_list: list of (page_num, rects) tuples.
    Returns (out_path, width, height, page_nums).
    """
    crops = []
    page_nums = []
    n_pages = len(page_rects_list)
    for idx, (page_num, rects) in enumerate(page_rects_list):
        # At page seams, use tight padding: less bottom on first crop, less top on continuation
        if n_pages == 1:
            crop = _render_one_crop(doc, page_num, rects)
        elif idx == 0:
            crop = _render_one_crop(doc, page_num, rects, pad_bottom=30)
        else:
            crop = _render_one_crop(doc, page_num, rects, pad_top=20)
        crops.append(crop)
        page_nums.append(page_num)

    if len(crops) == 1:
        img = crops[0]
    else:
        # Stitch vertically with a thin divider, keeping height compact
        gap = 4
        total_w = max(c.width for c in crops)
        total_h = sum(c.height for c in crops) + gap * (len(crops) - 1)
        img = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        y = 0
        for i, crop in enumerate(crops):
            img.paste(crop, (0, y))
            y += crop.height
            if i < len(crops) - 1:
                # Draw a subtle gray divider line
                for x in range(total_w):
                    for g in range(gap):
                        img.putpixel((x, y + g), (200, 200, 200))
                y += gap

    out_path = os.path.join(img_dir, filename)
    img.save(out_path, "PNG")
    return out_path, img.width, img.height, page_nums


# === STEP 4: Generate annotated HTML ===

def generate_annotated_html(html_path, output_path, quote_images):
    """Insert PDF screenshot images after each court-quote in the HTML."""
    with open(html_path, "r") as f:
        content = f.read()

    # For each quote, find its court-quote div and insert an image after it
    # Match on a snippet of the quote text to uniquely identify each div
    for qi in quote_images:
        ref_text = qi["ref"]
        quote_text = qi["text"]
        img_path = qi["img_path"]
        page_nums = qi["page_nums"]

        # Format page label
        if len(page_nums) == 1:
            page_label = f"PDF page {page_nums[0] + 1}"
        else:
            page_label = f"PDF pages {page_nums[0] + 1}&ndash;{page_nums[-1] + 1}"

        # Build the image HTML to insert
        img_html = (
            f'\n    <div class="pdf-evidence" style="margin: 1rem 0 1.5rem; '
            f'border: 1px solid #c5baa6; border-radius: 4px; overflow: hidden; '
            f'background: #fff;">'
            f'\n      <div style="background: #1a1a2e; color: #aaa; padding: 0.35rem 0.75rem; '
            f'font-family: \'Courier New\', monospace; font-size: 0.55rem; '
            f'letter-spacing: 0.05em; text-transform: uppercase;">'
            f'From the judgment &mdash; {page_label}</div>'
            f'\n      <img src="{img_path}" alt="Highlighted excerpt from judgment - {ref_text}" '
            f'style="width: 100%; display: block;" loading="lazy">'
            f'\n    </div>'
        )

        # Use first ~40 chars of quote text as unique identifier within the div
        snippet = re.escape(quote_text[:40])
        pattern = re.compile(
            r'(<div\s+class="court-quote">\s*' + snippet + r'.*?</div>)',
            re.DOTALL
        )
        match = pattern.search(content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + img_html + content[insert_pos:]
        else:
            print(f"  WARNING: Could not find insertion point for: {quote_text[:60]}...")

    # Update title to indicate annotated version
    content = content.replace(
        "<title>Discharged: The Delhi Excise Policy Case Explained</title>",
        "<title>Discharged: The Delhi Excise Policy Case Explained (Annotated)</title>"
    )

    # Add a small banner after the header indicating this is the annotated version
    header_end = content.find("</header>")
    if header_end != -1:
        banner = (
            '\n<div style="background: #b8860b; color: #fff; text-align: center; '
            'padding: 0.5rem; font-family: \'Courier New\', monospace; '
            'font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;">'
            'Annotated Edition &mdash; With highlighted excerpts from the original judgment PDF'
            '</div>'
        )
        content = content[:header_end + len("</header>")] + banner + content[header_end + len("</header>"):]

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


# === MAIN ===

def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    # Step 1: Extract quotes from HTML
    quotes = extract_quotes(HTML_PATH)

    # Step 2: Open PDF and build paragraph index
    doc = fitz.open(PDF_PATH)
    para_index = build_para_page_index(doc)

    # Step 3: Find, highlight, and render each quote
    print(f"\n[3/5] Finding and highlighting quotes in PDF...")
    quote_images = []
    for i, quote in enumerate(quotes):
        page_rects_list = find_and_highlight(doc, quote, para_index)
        if page_rects_list:
            filename = f"quote_{i+1:02d}.png"
            img_path, w, h, page_nums = highlight_and_render(doc, page_rects_list, filename, IMG_DIR)
            quote_images.append({
                "ref": quote["ref"],
                "text": quote["text"],
                "img_path": f"{IMG_DIR}/{filename}",
                "page_nums": page_nums,
                "width": w,
                "height": h,
            })
            pages_str = "–".join(str(p + 1) for p in page_nums)
            span_tag = " [SPANS PAGES]" if len(page_nums) > 1 else ""
            hl_count = sum(len(r) for _, r in page_rects_list)
            print(f"  [{i+1}] {quote['ref']}: page {pages_str} -> {w}x{h}px ({hl_count} highlights){span_tag}")
        else:
            print(f"  [{i+1}] {quote['ref']}: NOT FOUND in PDF")

    doc.close()

    # Step 4: Generate annotated HTML
    print(f"\n[4/5] Generating annotated HTML...")
    output = generate_annotated_html(HTML_PATH, OUTPUT_HTML, quote_images)
    print(f"  -> {output}")

    # Step 5: Summary
    found = len(quote_images)
    total = len(quotes)
    print(f"\n[5/5] Done: {found}/{total} quotes highlighted and embedded")
    print(f"  Images: {IMG_DIR}/")
    print(f"  Annotated blog: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
