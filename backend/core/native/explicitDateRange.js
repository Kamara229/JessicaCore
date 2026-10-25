/**
 * Parse an explicit Russian day/month range where only the final
 * endpoint carries a year: "с 27 февраля по 2 марта 2024 года".
 * Do not advance an excluded starting date: date_diff elapsed already
 * implements (start, end] and accounts for leap years.
 */
import { parseNativeDate } from "./dateUtils.js";

export function extractExplicitDateRange(value) {
    const text = String(value || "");
    const match = text.match(
        /(?:^|[^\p{L}\p{N}])(?:с|со|от)\s+(\d{1,2})\s+([а-яё]+)\s+(?:по|до)\s+(\d{1,2})\s+([а-яё]+)\s+(\d{4})(?:\s*г(?:ода|\.)?)?/iu
    );
    if (!match) return null;

    const end = parseNativeDate(`${match[3]} ${match[4]} ${match[5]}`);
    if (!end) return null;

    const sameYearStart = parseNativeDate(
        `${match[1]} ${match[2]} ${match[5]}`
    );
    if (!sameYearStart) return null;

    // December-to-January ranges refer to the preceding calendar year.
    const year = sameYearStart.date > end.date
        ? Number(match[5]) - 1 : Number(match[5]);
    const start = parseNativeDate(`${match[1]} ${match[2]} ${year}`);
    if (!start || start.date > end.date) return null;
    return {startDate:start.iso, endDate:end.iso};
}
