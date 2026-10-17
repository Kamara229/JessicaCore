/*
 * =========================================================
 * JESSICA PLANNER EVIDENCE POLICY v2
 * =========================================================
 *
 * User Task
 *      +
 * Execution Plan
 *          ↓
 * Required Evidence Strength
 *
 *
 * Назначение:
 *
 * не позволять Planner выбирать
 * слишком слабый Evidence Mode
 * для фактической задачи.
 *
 *
 * ВАЖНО:
 *
 * evidenceValidator.js
 *
 * проверяет:
 *
 * Plan internally consistent?
 *
 *
 * plannerEvidencePolicy.js
 *
 * проверяет:
 *
 * Evidence sufficiently strong
 * for the USER TASK?
 *
 * =========================================================
 */


/*
 * =========================================================
 * MODES
 * =========================================================
 */


const MODE_NONE =
    "none";


const MODE_SEARCH =
    "search_results";


const MODE_SOURCE =
    "source_content";


const MODE_WEIGHT = {

    [MODE_NONE]:
        0,

    [MODE_SEARCH]:
        1,

    [MODE_SOURCE]:
        2

};


/*
 * =========================================================
 * TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toLowerCase();

}


/*
 * =========================================================
 * MATCH
 * =========================================================
 */


function matchesAny(
    text,
    patterns
) {

    return patterns.some(
        pattern =>
            pattern.test(
                text
            )
    );

}


/*
 * =========================================================
 * EXPLICIT SOURCE CONTENT
 * =========================================================
 *
 * Пользователь прямо требует:
 *
 * - открыть страницу;
 * - проверить содержимое;
 * - подтвердить по источнику.
 *
 * =========================================================
 */


const EXPLICIT_SOURCE_PATTERNS = [

    /открой\p{L}*.*(страниц|сайт|источник|документ)/iu,

    /проверь\p{L}*.*содержим\p{L}*/iu,

    /по\s+содержим\p{L}*/iu,

    /подтверд\p{L}*.*(официальн|источник|страниц|сайт|документ)/iu,

    /прочитай\p{L}*.*(страниц|сайт|документ)/iu,

    /перейди\p{L}*.*(страниц|сайт|ссылк)/iu,

    /\bopen\b.*\b(page|source|website|document)\b/iu,

    /\bverify\b.*\b(source|page|content|official)\b/iu,

    /\bcheck\b.*\b(page|source|content)\b/iu

];


/*
 * =========================================================
 * FRESHNESS
 * =========================================================
 */


const FRESHNESS_PATTERNS = [

    /актуальн\p{L}*/iu,

    /текущ\p{L}*/iu,

    /последн\p{L}*/iu,

    /действующ\p{L}*/iu,

    /сейчас/iu,

    /на\s+сегодня/iu,

    /сегодняшн\p{L}*/iu,

    /новейш\p{L}*/iu,

    /\blatest\b/iu,

    /\bcurrent\b/iu,

    /\bcurrently\b/iu

];


/*
 * =========================================================
 * AUTHORITATIVE SOURCE
 * =========================================================
 */


const AUTHORITY_PATTERNS = [

    /официальн\p{L}*\s+(сайт|источник|страниц|документац)/iu,

    /на\s+официальн\p{L}*\s+сайт/iu,

    /с\s+официальн\p{L}*\s+(сайт|источник)/iu,

    /сайт\p{L}*\s+производител/iu,

    /документац\p{L}*\s+разработчик/iu,

    /\bofficial\s+(site|website|source|page|documentation)\b/iu

];


/*
 * =========================================================
 * EXACT FACT
 * =========================================================
 *
 * Факты, для которых search snippet
 * не должен считаться сильным Evidence,
 * если значение актуальное/меняющееся.
 *
 * =========================================================
 */


const EXACT_FACT_PATTERNS = [

    /верси\p{L}*/iu,

    /релиз\p{L}*/iu,

    /\brelease\b/iu,

    /\bversion\b/iu,

    /дат\p{L}*\s+(выпуск|релиз|публикац|обновлен)/iu,

    /номер\p{L}*\s+верси/iu,

    /цен\p{L}*/iu,

    /стоимост\p{L}*/iu,

    /тариф\p{L}*/iu,

    /статус\p{L}*/iu,

    /услови\p{L}*/iu,

    /срок\p{L}*/iu,

    /\brate\b/iu,

    /\bprice\b/iu,

    /\bstatus\b/iu

];


/*
 * =========================================================
 * STABLE SOFTWARE VERSION
 * =========================================================
 */


const STABLE_VERSION_PATTERNS = [

    /стабильн\p{L}*\s+верси\p{L}*/iu,

    /\bstable\s+version\b/iu,

    /\blts\b/iu

];


/*
 * =========================================================
 * REQUIRED MODE
 * =========================================================
 */


export function resolveRequiredEvidenceMode(
    task
) {

    const text =

        normalizeText(
            task
        );


    if(
        !text
    ){

        return {

            mode:
                MODE_NONE,

            reason:
                "Task отсутствует",

            signals: {}

        };

    }


    const explicitSource =

        matchesAny(

            text,

            EXPLICIT_SOURCE_PATTERNS

        );


    const freshness =

        matchesAny(

            text,

            FRESHNESS_PATTERNS

        );


    const authority =

        matchesAny(

            text,

            AUTHORITY_PATTERNS

        );


    const exactFact =

        matchesAny(

            text,

            EXACT_FACT_PATTERNS

        );


    const stableVersion =

        matchesAny(

            text,

            STABLE_VERSION_PATTERNS

        );


    /*
     * =====================================================
     * SOURCE CONTENT REQUIRED
     * =====================================================
     */


    if(
        explicitSource
    ){

        return {

            mode:
                MODE_SOURCE,

            reason:
                "Пользователь явно требует проверку содержимого конкретного внешнего источника",

            signals: {

                explicitSource,

                freshness,

                authority,

                exactFact,

                stableVersion

            }

        };

    }


    /*
     * Актуальный конкретный факт.
     *
     * Например:
     *
     * актуальная версия;
     * текущая цена;
     * действующий тариф;
     * последняя стабильная версия.
     */


    if(
        freshness
        &&
        (
            exactFact
            ||
            stableVersion
        )
    ){

        return {

            mode:
                MODE_SOURCE,

            reason:
                "Актуальный изменяемый факт требует подтверждения содержимым источника",

            signals: {

                explicitSource,

                freshness,

                authority,

                exactFact,

                stableVersion

            }

        };

    }


    /*
     * Официальный источник +
     * конкретный проверяемый факт.
     */


    if(
        authority
        &&
        (
            exactFact
            ||
            stableVersion
        )
    ){

        return {

            mode:
                MODE_SOURCE,

            reason:
                "Факт должен быть подтверждён непосредственно официальным источником",

            signals: {

                explicitSource,

                freshness,

                authority,

                exactFact,

                stableVersion

            }

        };

    }


    /*
     * =====================================================
     * SEARCH RESULTS
     * =====================================================
     *
     * Если задача требует внешнего поиска,
     * но нет требования читать страницу,
     * сам Planner всё ещё может выбрать
     * search_results.
     *
     * Policy здесь его не усиливает.
     *
     * =====================================================
     */


    return {

        mode:
            MODE_NONE,

        reason:
            "Evidence escalation не требуется",

        signals: {

            explicitSource,

            freshness,

            authority,

            exactFact,

            stableVersion

        }

    };

}


/*
 * =========================================================
 * VALIDATE POLICY
 * =========================================================
 */


export function validatePlannerEvidencePolicy({

    task,

    plan

} = {}) {

    const requirement =

        resolveRequiredEvidenceMode(
            task
        );


    /*
     * MODE_NONE означает:
     *
     * policy не предъявляет
     * дополнительного требования.
     *
     * Это НЕ означает,
     * что Planner обязан использовать none.
     */


    if(
        requirement.mode ===
        MODE_NONE
    ){

        return {

            success:
                true,

            requiredMode:
                MODE_NONE,

            actualMode:

                plan?.evidence?.mode

                ||

                MODE_NONE,

            reason:
                requirement.reason,

            signals:
                requirement.signals

        };

    }


    const actualMode =

        String(

            plan?.evidence?.mode

            ||

            MODE_NONE

        )
            .trim()
            .toLowerCase();


    const requiredWeight =

        MODE_WEIGHT[
            requirement.mode
        ]

        ??

        0;


    const actualWeight =

        MODE_WEIGHT[
            actualMode
        ]

        ??

        -1;


    if(
        actualWeight <
        requiredWeight
    ){

        return {

            success:
                false,

            requiredMode:
                requirement.mode,

            actualMode,

            signals:
                requirement.signals,

            text:

                (
                    `Недостаточный evidence.mode=${actualMode}. `
                    +
                    `Для этой задачи требуется evidence.mode=${requirement.mode}. `
                    +
                    requirement.reason
                    +
                    ". Добавь получение и проверку содержимого источника через web_fetch."
                )

        };

    }


    return {

        success:
            true,

        requiredMode:
            requirement.mode,

        actualMode,

        reason:
            requirement.reason,

        signals:
            requirement.signals

    };

      }
