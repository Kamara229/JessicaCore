// Ensure immutable snapshots form a complete v1..vN sequence.
export function requireContiguousExperienceHistory(history) {
    if (!Array.isArray(history)) {
        throw new Error("Experience history is not an array");
    }
    const versions = new Set();
    for (const item of history) {
        const version = Number(item?.version);
        if (!Number.isSafeInteger(version) || version < 1 || versions.has(version)) {
            throw new Error("Invalid or duplicate Experience history version");
        }
        versions.add(version);
    }
    if (versions.size > 0 && Math.max(...versions) !== versions.size) {
        throw new Error("Experience history has a missing version");
    }
    return versions.size;
}
