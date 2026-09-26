/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER
 * =========================================================
 *
 * Анализирует результат выполнения Jessica
 * и определяет:
 *
 * - является ли опыт переиспользуемым;
 * - есть ли кандидат на новый Skill;
 * - нужно ли запускать обучение.
 *
 *
 * НЕ:
 *
 * - сохраняет в Supabase;
 * - создаёт Skill;
 * - изменяет Experience.
 *
 * Только анализ.
 *
 * =========================================================
 */


/*
 * =========================================================
 * KEYWORDS
 * =========================================================
 *
 * Базовые признаки повторяемых задач.
 *
 * В будущем можно заменить
 * на AI Pattern Detector.
 *
 * =========================================================
 */


const REUSABLE_PATTERNS = [

    {
        id:
            "company-contact-finder",

        keywords:[

            "контакт",
            "почта",
            "email",
            "телефон",
            "адрес компании",
            "support",
            "contact"

        ],

        category:
            "web-research"

    },


    {
        id:
            "official-website-verification",

        keywords:[

            "официальный сайт",
            "official website",
            "домен",
            "website"

        ],

        category:
            "verification"

    }


];




/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    text
) {

    return String(
        text || ""
    )
        .toLowerCase()
        .trim();

}





/*
 * =========================================================
 * FIND PATTERN
 * =========================================================
 */


function detectPattern(
    task
) {


    const text =
        normalizeText(
            task
        );



    let best =
        null;



    let score =
        0;



    for (
        const pattern
        of REUSABLE_PATTERNS
    ) {


        const matches =
            pattern.keywords.filter(

                keyword =>
                    text.includes(
                        keyword
                    )

            ).length;



        if (
            matches > score
        ) {

            score =
                matches;


            best =
                pattern;

        }

    }



    if (
        !best
    ) {

        return null;

    }



    return {

        skillId:
            best.id,


        category:
            best.category,


        matches:
            score,


        confidence:
            Math.min(
                score / 5,
                1
            )

    };

}





/*
 * =========================================================
 * ANALYZE TRACE
 * =========================================================
 */


export function analyzeExecutionTrace(
    trace
) {


    if (
        !trace ||
        typeof trace !== "object"
    ) {

        return {

            reusable:
                false,

            reason:
                "Нет Execution Trace",

            skillCandidate:
                null

        };

    }



    /*
     * =====================================================
     * CHECK RESULT
     * =====================================================
     */


    const completed =
        trace.stats?.completed || 0;



    if (
        completed === 0
    ) {

        return {

            reusable:
                false,

            reason:
                "Нет успешного выполнения",

            skillCandidate:
                null

        };

    }




    /*
     * =====================================================
     * DETECT PATTERN
     * =====================================================
     */


    const pattern =
        detectPattern(
            trace.task
        );



    if (
        !pattern
    ) {

        return {

            reusable:
                false,

            reason:
                "Повторяемый паттерн не найден",

            skillCandidate:
                null

        };

    }





    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {


        reusable:
            true,


        reason:
            "Обнаружен повторяемый сценарий",



        skillCandidate: {


            skillId:
                pattern.skillId,


            category:
                pattern.category,


            confidence:
                pattern.confidence,


            source:
                "execution-trace"


        }


    };


}
