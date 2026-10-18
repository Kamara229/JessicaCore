import { extractDates } from "./releaseFactDates.js";

export function buildEvidenceFragment(sectionText, candidate) {
    if (!candidate) return "";

    const firstIndex = Math.min(candidate.index, candidate.dateIndex);
    const lastIndex = Math.max(
        candidate.index + candidate.version.length,
        candidate.dateIndex + String(candidate.rawDate || "").length
    );

    const start = Math.max(0, firstIndex - 220);
    const end = Math.min(sectionText.length, lastIndex + 420);

    return sectionText
        .slice(start, end)
        .replace(/\s+/g, " ")
        .trim();
}

export function verifyEvidence(fragment, candidate) {
    if (!fragment || !fragment.includes(candidate.version)) {
        return false;
    }

    return extractDates(fragment)
        .some(date => date.value === candidate.releaseDate);
}
