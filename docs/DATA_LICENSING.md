# Data sources and licensing

*Research and adversarial verification: 2026-11-18. This is an engineering assessment to guide data decisions, not legal advice.
The machine-enforced record is `data/manifests/sources.yaml`. Every source decision there cites a URL
and an exact quote.*

## Policy

1. Training text must be **verifiably** public domain, or licensed for ML training, distribution of
   trained weights and a public demo. "Publicly accessible", "free API" or "on GitHub" is not a licence.
2. **BBT / Prabhupāda translations and purports are excluded** unless written permission covering
   those uses is obtained.
3. Non-expressive **facts** (chapter/verse structure, verse counts, speaker labels) may be derived from
   sources marked `usage.metadata`.
4. Translations are never mixed anonymously. Every record carries its `source_id`, and training text
   names its translation.
5. Raw and processed texts are not committed. They are reproducible from pinned URLs and checksums.

## Copyright terms used in the analysis

| Jurisdiction | Rule applied |
|---|---|
| India | Copyright Act 1957: author's life + 60 years (counted from the year after death) |
| United States | Works published abroad before 1931 are public domain (as of 2026). Later foreign works were restored by the URAA (1996-01-01) if still protected at home; they then run 95 years from publication. |
| EU | Life + 70 years. For non-EU works the shorter term of the country of origin applies (rule of the shorter term). |

## Sanskrit sources

| Source | Licence evidence | Finding | Decision |
|---|---|---|---|
| **GRETIL** `sa_bhAgavatapurANa.xml` | TEI header: *"Distributed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License."* Legacy file: *"FOR REFERENCE PURPOSES ONLY! COPYRIGHT AND TERMS OF USAGE AS FOR SOURCE FILE."* | Complete: 12 cantos, 335 chapters, 14,099 records. The printed source edition is not named. Cantos 3, 4, 5, 10, 11 and 12 (~67 %) use BBT house style (hyphenated compounds, *śrī-śuka uvāca*) and contain OCR errors typical of a printed romanised edition. GRETIL's named download source writes: *"I believe the original source of this text was the BBT Folio version of the Srimad Bhagavatam."* | **Metadata only** (verse structure, speaker labels). Not training text. |
| **DCS** (Digital Corpus of Sanskrit) | *"licensed under the Creative Common BY 4.0 (CC BY 4.0) license"* | 119 of 335 chapters; the text is derived from GRETIL, so chain of title is unclear | under review |
| **Sanskrit Wikisource** | site footer: CC BY-SA 4.0 | complete, standard numbering; source edition undocumented; readings differ from GRETIL | under review (best complete candidate) |
| **sanskritdocuments.org** | *"…ingestion of these formatted files for external websites, applications, or AI models is strictly prohibited without prior written permission."* | GRETIL family text | **excluded** |
| **Gaudiya Grantha Mandira** | "© Jiva Gaudiya Grantha Mandira"; editor believes the text came from the BBT Folio | | **excluded** |
| Burnouf et al., Paris 1840–98 | public domain by date | Devanagari text, but the existing OCR has no Devanagari | possible future source (needs re-OCR) |

## English translations

Every source below was researched and then **adversarially verified** by an independent reviewer
instructed to refute its claimed status (2026-11-18). The verdicts and their conditions are recorded
in the manifest.

### M. N. Dutt, *A Prose English Translation of Srimadbhagavatam* (Calcutta, Elysium Press, 1895–96): **public domain everywhere**

- Published in 1895–96, so public domain in the US. The translator died in 1912, so the work has been
  public domain in India since 1973 and in life+70 jurisdictions since 1983. Verdict: *usable with
  conditions*. The reviewer noted that Dutt's birth and death years are disputed, but the conclusion
  rests on the publication date either way.
- Structure: Books I–XII with numbered chapters, chapter summaries and speaker tags. The text is
  continuous prose **without verse numbers**; the introduction says "I have not considered it necessary
  to put in the numbers of the Slokas". It aligns reliably at chapter level only.
- Digitized copies:
  - archive.org has Books I–IX plus Book X chapters 1–5 (~300k words, good OCR). These are pinned by
    SHA-256 in the manifest and downloaded by the pipeline.
  - **Book X chapters 6–90, XI and XII exist only in the Google Books full-view scan** (`RQQ-AQAAMAAJ`,
    volumes 8–12). Google's usage guidelines forbid automated querying, so that PDF must be downloaded
    once by hand in a browser.
  - The full text is roughly 480–520k words.
- Conditions:
  - Use only the 1895–96 scans; later reprints may add copyrightable material.
  - The Asiatic Society scan (`india.history.resource.40625`) has an **unrelated 1880s pamphlet bound
    in** (Taleyarkhan, on industrial education). It must be removed, and the scan's pages re-ordered.
  - Never redistribute the scan files; publish only cleaned, derived text with provenance.
  - Credit: "Manmatha Nath Dutt, *A Prose English Translation of Srimadbhagavatam*, Calcutta 1895–96".

### S. Subba Rau, *Srimad Bhagavatam translated into easy English prose* (Tirupati, 1928): public domain **in the US only**

- Verse-numbered translation of Cantos 1–7, following Śrīdhara's commentary, with Viśiṣṭādvaita and
  Dvaita readings in footnotes. Volume II (Cantos 8–12) was not found digitally.
- US: public domain, since any URAA term ended on 2023-12-31. India and the EU: **unverified**. The
  translator's death year is unknown, and he may still have been active in 1937.
- Valuable for verse-level alignment, but not cleared for publicly released weights until the death
  year is established.

### Mohendra Nath Chatterjee, *The Çrīmadbhāgavatam* (Calcutta, 1895): public domain **in the US only**

- Canto 1 has Devanagari, a verse-numbered translation and extensive notes; Canto 2 exists at HathiTrust.
- HathiTrust's rights codes are inconsistent across copies ("pd" vs "pdus"). The translator's death
  year is unverified, so worldwide status rests on presumption.
- The notes quote third-party translations, which must be separated from the verse text.

### J. M. Sanyal, *The Srimad-Bhagbatam* (Calcutta, 1929–c.1936): **blocked**

- A complete prose translation with verse-range markers. The translator is unidentified (the LC
  authority record has no dates), so India and EU terms cannot be computed.
- In the US only the 1929–30 parts are public domain; the rest may be URAA-restored until 2027–2032.
- A 1954 scholarly reference calls it "the very free recent paraphrase of Sanyal". Later editions are
  excluded.

### Not usable

| Source | Finding |
|---|---|
| **A. C. Bhaktivedanta Swami Prabhupāda (BBT)** translation and purports | BBTI: "The Bhaktivedanta Book Trust (BBT) is a California religious trust that owns the copyrights to the works published by His Divine Grace A.C. Bhaktivedanta Swami Prabhupada during his lifetime." vedabase: "Content used with permission of © The Bhaktivedanta Book Trust International, Inc. All rights reserved." **Excluded**; never downloaded. |
| **bhagavata.org = srimadbhagavatam.org** (Anand Aadhar, "The Story of the Fortunate One") | Same site under two domains. CC BY-NC-SA 3.0 is declared, but the translator states the work is based on Prabhupāda's BBT translation, so the grant cannot cover the underlying expression. **Permission required.** |
| ~20 GitHub / Hugging Face "Bhagavatam" datasets and APIs | Scraped from vedabase (BBT); one maintainer writes "I haven't took any written permission from respective organisations". Repository licences (MIT, Apache-2.0, GPL-3.0) grant nothing in BBT text. **Excluded.** |
| Gita Press (1971), Tagare (1976–78), Tapasyananda (1980s), Debroy (2018–19), Prabhavananda (1943, US renewal R491893 in 1970) | In copyright. **Permission required.** |
| Eastwick (1851) and Hollings (1848), *Prem Sagar* | Public domain everywhere, but English renderings of a Hindi retelling of Canto 10, not the Bhāgavatam. Eastwick's preface contains anti-Hindu polemic. **Not used.** |
| P. N. Sinha, *A Study of the Bhāgavata Purāṇa* (1901); *Bhaktiratnāvalī* (1912) | Interpretive or anthological, not translations. Kept under review; if ever used, they would be labelled as interpretation. |

### What this means for training

- The first trainable English corpus will be **Dutt alone**, once the Phase 1b parser exists and the
  Google Books volume has been added.
- **Style:** Dutt's English is 19th-century and sometimes free. SFT answers will be written in modern
  English and attribute wording to the translation.
- **Citations:** without verse numbers, the model is trained to cite at chapter level ("SB 10.29"). Verse-level citations
  (e.g. via Subba Rau's verse numbering for Cantos 1–7) depend on resolving the US-only status, which is
  a decision for the project owner.

## Re-OCR and cleaning plan for public-domain scans (Phase 1b)

1. Fetch page images or PDFs of the cleared edition from archive.org (licence-gated `bhagavatam data
   download`, with checksums).
2. OCR locally: Apple Vision on the Mac, which is free and accurate, or Tesseract as a portable
   fallback. Keep the raw OCR as an artefact.
3. Clean with `data/normalize/ocr.py`: running heads, page numbers, de-hyphenation using corpus
   evidence, ligatures. Ambiguous cases go to a review list.
4. Segment Book/Chapter and validate against the 335-chapter structure.
5. Human spot-check per chapter. Corrections go into an overlay file, so raw data is never edited
   in place.
