/**
 * Keep substantive document findings ahead of boilerplate for broad summaries.
 * User-requested contact, author and registration details remain available.
 * This is a relevance filter, not an entailment or legal-validity verifier.
 */
const HEADER_DETAILS = /(?:телефон|контактн|факс|e.?mail|электронн[а-я]*\s+почт|реквизит|при[её]мная|подпис[ьи]|названи[ея]\s+документ|носит\s+названи|текст\s+адресован|документ\s+завершается|упоминается\s+(?:университет|институт|организац))/iu;
const REQUEST_FOR_DETAILS = /(?:телефон|контакт|факс|email|e-mail|почт|реквизит|подпис|кто\s+автор|кому\s+адресован|получател|отправител|названи[ея]|шапк[аеу]|выходн[ые]\s+данн)/iu;
const SUBSTANCE = /(?:предлага|измен|замен|корректир|влеч[её]т|позвол|не\s+соответств|не\s+удовлетвор|расход|стоимост|цен[аы]|смет[аыуе]|срок|эконом|требован|согласован|обязательств|решени[ея]|результат|заключени[ея]|рис[кч]|повыш|уменьш|увелич)/iu;

export function selectSalientFindings(grounded, task, limit = 14) {
    if (!Array.isArray(grounded) || grounded.length === 0) return [];
    const userWantsHeaders = REQUEST_FOR_DETAILS.test(String(task || ""));
    const substantive = grounded.filter(item => SUBSTANCE.test(item.text));
    // Never discard everything merely because of an imperfect heuristic.
    if (userWantsHeaders || substantive.length < 2) return grounded.slice(0, limit);
    const substantiveAndDetails = grounded.filter(item => !HEADER_DETAILS.test(item.text));
    return (substantiveAndDetails.length ? substantiveAndDetails : grounded).slice(0, limit);
}
