"""
Prepare the judgment text as JSON input for DocETL pipeline.
Splits the 549-page judgment into page-based sections for processing.
"""
import json
import re
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "judgment_full_text.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "judgment_sections.json")

def split_into_pages(text):
    """Split on the page markers in the judgment text."""
    # The PDF has page markers like "CBI Vs. Kuldeep Singh & others   Page no. X of 549"
    pages = re.split(r'\s*CBI Vs\. Kuldeep Singh & others\s+Page no\. \d+ of 549\s*', text)
    return [p.strip() for p in pages if p.strip()]

def group_pages(pages, group_size=10):
    """Group pages into sections of group_size pages each."""
    sections = []
    for i in range(0, len(pages), group_size):
        chunk = pages[i:i+group_size]
        start_page = i + 1
        end_page = min(i + group_size, len(pages))
        sections.append({
            "section_id": f"pages_{start_page:03d}_{end_page:03d}",
            "start_page": start_page,
            "end_page": end_page,
            "text": "\n\n".join(chunk),
            "case_name": "CBI vs. Kuldeep Singh & Others",
            "case_number": "CBI Case No. 56/2022",
            "court": "Special Court (PC Act) CBI-23, Rouse Avenue, New Delhi",
            "judge": "Sh. Jitendra Singh",
            "order_date": "27.02.2026",
            "total_pages": 549
        })
    return sections

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    pages = split_into_pages(text)
    print(f"Total pages extracted: {len(pages)}")

    sections = group_pages(pages, group_size=10)
    print(f"Total sections created: {len(sections)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    # Print section sizes for debugging
    for s in sections:
        chars = len(s["text"])
        words = len(s["text"].split())
        print(f"  {s['section_id']}: {chars:,} chars, ~{words:,} words")

    print(f"\nOutput written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
