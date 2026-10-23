/**
 * Formats an already calculated calendar interval without changing arithmetic.
 * For elapsed mode, (start, end] excludes the first date and includes the last.
 */
export function formatDateDifference(start, end, days, mode) {
    if (!Number.isSafeInteger(days) || days < 0) throw Error("invalid-day-interval");
    if (mode === "elapsed") return (
        "От " + start + " до " + end + " прошло " + days +
        " полных календарных дней. Начальная дата " + start +
        " не учитывается, конечная дата " + end + " учитывается."
    );
    return "Между " + start + " и " + end + " " + days + " полных дней.";
}
