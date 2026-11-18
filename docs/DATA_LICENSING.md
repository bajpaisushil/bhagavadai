# Data sources and licensing

*Research date 2026-11-18. This is an engineering assessment to guide data decisions, not legal advice.
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

### M. N. Dutt, *A Prose English Translation of Srimadbhagabatam* (Calcutta, Elysium Press, 1895–96)

- Title page (seen in the archive.org OCR): *"A PROSE ENGLISH TRANSLATION OF SRIMADBHAGABATAM. BOOK I.
  EDITED AND PUBLISHED BY MANMATHA NATH DUTT, M.A., M.R.A.S. … CALCUTTA: Printed by H. C. Dass, Elysium
  Press … 1895"* (the OCR misreads the date as "1805").
- **Status: leading candidate; verification in progress.** Published 1895–96, so it is public domain
  in the US by date. Public-domain status in India and the EU depends on the translator's death year,
  which is pending documentary confirmation. The completeness of the 12 books, and the OCR quality of
  each item, are also being checked.
- Structure: Book → Chapter headings, with a descriptive table of contents. **Continuous prose without
  verse numbers.** Alignment is therefore reliable at chapter level only. Verse-level citations require
  a separately verified alignment.

### J. M. Sanyal, *The Srimad-Bhagbatam of Krishna-Dwaipayana Vyasa* (Calcutta, 1929–c.1936, 26 parts)

- Complete English prose translation of all 12 books, without Sanskrit text or commentary. Each
  paragraph ends with a verse-range marker such as `(21-23)`, so it can be aligned to verse ranges.
- **The translator is unidentified.** The LC name authority gives "Sanyal, J. M." with no dates, and
  no full name or death year was found in any authority file or in the books. The Indian and EU terms
  therefore cannot be computed.
- US status: parts published in 1929–30 (Books I–II and the start of III) are public domain. The parts
  from 1931–36 are public domain only if the translator died by 1940. Otherwise they were URAA-restored
  and stay protected until 2027–2032.
- Later editions (2nd ed. 1952–54, 1964–65, Munshiram Manoharlal 1970–2000) add revisions and an index
  of unknown authorship.
- HathiTrust marks every record "ic" (search-only), but by automatic bibliographic rule, not review.
- A 1954 scholarly reference describes it as *"the very free recent paraphrase of Sanyal"*.
- **Decision: not used** until the translator can be identified. Later editions are excluded.

### Other English sources

| Source | Status |
|---|---|
| A. C. Bhaktivedanta Swami Prabhupāda, translation and purports (BBT) | © The Bhaktivedanta Book Trust. **Excluded** (no permission covering ML training, weights or demo). |
| Gita Press (C. L. Goswami & M. A. Sastri, 1971) | in copyright: excluded |
| G. V. Tagare (Motilal Banarsidass, 1976–78) | in copyright: excluded |
| Swami Tapasyananda (Ramakrishna Math) | in copyright: excluded |
| Bibek Debroy (Penguin, 2018) | in copyright: excluded |
| P. N. Sinha, *A Study of the Bhāgavata Purāṇa* (1901, Project Gutenberg #39442) | a study with interpretation; status being verified. If used, it is always labelled as interpretation. |
| Websites (bhagavata.org, srimadbhagavatam.org) and structured datasets on GitHub/Hugging Face | being verified; datasets scraped from vedabase are BBT text regardless of any repository licence |

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
