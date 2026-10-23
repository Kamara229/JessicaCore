/** Russian day-count inflection; pure and independent of date arithmetic. */
export function russianDayForm(count) {
    if (!Number.isSafeInteger(count) || count < 0) throw Error("invalid-day-count");
    const lastTwo = count % 100;
    const last = count % 10;
    if (lastTwo < 11 || lastTwo > 14) {
        if (last === 1) return {verb:"прошёл",full:"полный календарный день",short:"полный день"};
        if (last >= 2 && last <= 4) return {verb:"прошло",full:"полных календарных дня",short:"полных дня"};
    }
    return {verb:"прошло",full:"полных календарных дней",short:"полных дней"};
}
