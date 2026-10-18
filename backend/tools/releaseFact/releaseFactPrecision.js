/*
 * RELEASE FACT PRECISION GUARD
 *
 * A major/minor Python branch (3.15) is not a complete
 * distributable release (3.15.0). Never silently add ".0".
 */
export function assessReleasePrecision({subject, version}) {
    const product = String(subject || "").toLowerCase();
    const parts = String(version || "").split(".");
    const exactReleaseRequested =
        /(последн|актуальн|стабильн|релиз|выпуск|latest|current|stable|release)/iu
            .test(product);

    if (/\bpython\b/iu.test(product) && exactReleaseRequested
        && parts.length < 3) {
        return {
            precise: false,
            reason: "Указана ветка Python, а не точный релиз. Нужна официальная страница релиза с полным номером версии и датой."
        };
    }

    return { precise: true };
}
