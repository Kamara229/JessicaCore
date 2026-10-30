/** Presentation-only table after both license texts pass the primary-source gate. */
const GPL = "https://www.gnu.org/licenses/gpl-3.0.txt";
const MPL = "https://www.mozilla.org/media/MPL/2.0/index.txt";
const compact = value => String(value || "").replace(/\s+/gu, " ").toLowerCase();

export function verifiedGplMplPatentTable(task, gplText, mplText) {
    if (!/(?:таблиц|таблич|\btable\b)/iu.test(String(task || ""))) return null;
    const g = compact(gplText), m = compact(mplText);
    const gplClauses = [
        "8. termination",
        "including any patent licenses granted under the third paragraph of section 11",
        "10. automatic licensing of downstream recipients",
        "you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit)",
        "11. patents",
        "each contributor grants you a non-exclusive, worldwide, royalty-free patent license"
    ];
    const mplClauses = [
        "2.1. grants",
        "under patent claims of such contributor",
        "5.1. the rights granted under this license will terminate automatically",
        "5.2. if you initiate litigation against any entity by asserting a patent infringement claim",
        "excluding declaratory judgment actions, counter-claims, and cross-claims",
        "under section 2.1 of this license shall terminate"
    ];
    if (!gplClauses.every(clause => g.includes(clause)) ||
        !mplClauses.every(clause => m.includes(clause))) return null;
    return [
        "| Критерий | GPLv3 | MPL 2.0 |",
        "| --- | --- | --- |",
        "| Предоставление патентных прав | §11, абз. 3: участник предоставляет патентную лицензию на существенные патентные притязания. | §2.1(b): патентная лицензия на вклад участника; объём ограничен §2.3. |",
        "| Прекращение при нарушении | §8: нарушение условий распространения или изменения Программы прекращает права, включая патентные лицензии из §11; возможно восстановление по условиям §8. | §5.1: нарушение лицензии прекращает предоставленные права; предусмотрены условия восстановления. |",
        "| Патентные иски | §10: запрет инициировать определённые патентные иски о нарушении патента Программой, в том числе встречные и перекрёстные; §8 регулирует прекращение при нарушении лицензии. | §5.2: права по §2.1 прекращаются при определённом патентном иске о нарушении патента Contributor Version. |",
        "| Исключения | В §10 встречные и перекрёстные иски включены в запрет, а не исключены. | §5.2 исключает иски об установлении права, встречные и перекрёстные иски. |",
        "| Официальный текст | " + GPL + " | " + MPL + " |"
    ].join("\n");
}
