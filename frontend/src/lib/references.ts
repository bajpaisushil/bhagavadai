// Mirrors bhagavatam_ai.data.reference.find_references: an explicit prefix is required so that dates
// or section numbers are not mistaken for verse citations.
const REFERENCE = /\b(?:SB|Bhāg\.|Bhag\.|BhP|Bhāgavatam|Bhagavatam)\s*\d+\.\d+\.\d+(?:\s*[-–]\s*\d+)?(?!\d)/g;

export type Segment = { kind: "text"; text: string } | { kind: "ref"; text: string };

export function splitReferences(text: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(REFERENCE)) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: text.slice(last, start) });
    segments.push({ kind: "ref", text: match[0] });
    last = start + match[0].length;
  }
  if (last < text.length) segments.push({ kind: "text", text: text.slice(last) });
  return segments;
}
