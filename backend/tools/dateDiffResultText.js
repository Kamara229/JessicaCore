/**
 * Formats an already calculated calendar interval without changing arithmetic.
 * For elapsed mode, (start, end] excludes the first date and includes the last.
 */
import {russianDayForm} from "./russianDayInflection.js";
import {countedCalendarDays} from "./dateDiffCountedDays.js";

export function formatDateDifference(start, end, days, mode) {
    if (!Number.isSafeInteger(days) || days < 0) throw Error("invalid-day-interval");
    const form = russianDayForm(days);
    if (mode === "elapsed") return (
        "От " + start + " до " + end + " " + form.verb + " " + days +
        " " + form.full + ". Начальная дата " + start +
        " не учитывается, конечная дата " + end + " учитывается." +
        countedCalendarDays(start, end, days)
    );
    return "Между " + start + " и " + end + " " + days + " " + form.short + ".";
}
