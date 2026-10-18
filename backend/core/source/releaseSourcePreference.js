/*
 * Prefer official release/download listings over branch lifecycle tables
 * when the user asks for an exact latest Python release.
 * A specific numbered release URL may be stale, so only boost listings.
 */
export function scoreReleaseSourcePreference(task, candidate) {
    const request = String(task || "");
    if (!/\bpython\b/iu.test(request)
        || !/(последн|актуальн|стабильн|релиз|выпуск|latest|stable|release)/iu.test(request)) {
        return 0;
    }

    let url;
    try {
        url = new URL(String(candidate?.url || ""));
    } catch {
        return 0;
    }

    const host = url.hostname.toLowerCase();
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const pythonOrg = host === "python.org" || host === "www.python.org";
    if (pythonOrg && (path === "/downloads" || path === "/downloads/windows"
        || path === "/downloads/macos")) return 30;
    if (pythonOrg && path.startsWith("/downloads/release/")) return -8;
    if (host === "devguide.python.org" && path.startsWith("/versions")) return -24;
    return 0;
}
