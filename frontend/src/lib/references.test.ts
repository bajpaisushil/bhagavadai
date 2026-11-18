import { describe, expect, it } from "vitest";

import { splitReferences } from "./references";

describe("splitReferences", () => {
  it("finds explicit verse citations, including ranges", () => {
    const segments = splitReferences("See SB 1.2.6 and Bhāgavatam 10.14.58-60, not 2026.11.18.");
    expect(segments.filter((s) => s.kind === "ref").map((s) => s.text)).toEqual(["SB 1.2.6", "Bhāgavatam 10.14.58-60"]);
    expect(segments.map((s) => s.text).join("")).toBe("See SB 1.2.6 and Bhāgavatam 10.14.58-60, not 2026.11.18.");
  });

  it("captures whole numbers instead of truncating them into valid-looking references", () => {
    // "SB 1.1.1000" must not become "SB 1.1.100"; the server decides whether the verse exists.
    expect(splitReferences("SB 1.1.1000").filter((s) => s.kind === "ref")).toEqual([{ kind: "ref", text: "SB 1.1.1000" }]);
  });

  it("returns plain text unchanged", () => {
    expect(splitReferences("no citations")).toEqual([{ kind: "text", text: "no citations" }]);
  });
});
