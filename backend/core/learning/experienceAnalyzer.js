/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER
 * =========================================================
 *
 * Анализирует Execution Trace.
 *
 *
 * Определяет:
 *
 * - улучшать существующий Skill;
 * - создать новый Skill;
 * - игнорировать опыт.
 *
 *
 * НЕ:
 *
 * - сохраняет Experience;
 * - пишет в Supabase;
 * - создаёт Skill.
 *
 * Только анализ.
 *
 * =========================================================
 */







/*
 * =========================================================
 * REUSABLE PATTERNS
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
            "official-source-verification",


        keywords:[

            "официальный сайт",

            "официальный источник",

            "документ",

            "сертификат",

            "проверить"

        ],


        category:
            "verification"

    }


];








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
    .toLowerCase()
    .trim();

}









/*
 * =========================================================
 * DETECT PATTERN
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

            )
            .length;




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



        confidence:

            Math.min(
                score / 5,
                1
            )

    };


}









/*
 * =========================================================
 * EXPERIENCE USAGE
 * =========================================================
 */


function analyzeExistingSkill(
    trace
) {


    const usage =
        trace?.experienceUsage;



    if (
        !usage ||
        usage.used !== true
    ) {

        return null;

    }





    if (
        !Array.isArray(
            usage.skills
        )
        ||
        usage.skills.length === 0
    ) {

        return null;

    }







    return {


        action:

            "SKILL_IMPROVEMENT",



        reason:

            "Использованный Skill дал новый успешный опыт",



        skillCandidate:


        {


            skills:

                usage.skills,



            source:

                "execution-trace",



            confidence:

                0.9


        }


    };


}









/*
 * =========================================================
 * NEW SKILL DETECTION
 * =========================================================
 */


function analyzeNewSkill(
    trace
) {


    const pattern =

        detectPattern(
            trace.task
        );



    if (
        !pattern
    ) {


        return null;

    }





    return {


        action:

            "NEW_SKILL",



        reason:

            "Обнаружен повторяемый сценарий без существующего Skill",



        skillCandidate:


        {

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









/*
 * =========================================================
 * MAIN ANALYSIS
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


            action:

                "IGNORE",



            reusable:

                false,



            reason:

                "Нет Execution Trace",



            skillCandidate:

                null


        };

    }







    /*
     * Только успешные выполнения
     */


    if (
        (trace.stats?.completed || 0) === 0
    ) {


        return {


            action:

                "IGNORE",



            reusable:

                false,



            reason:

                "Нет успешного результата",



            skillCandidate:

                null


        };


    }









    /*
     * Сначала проверяем:
     *
     * был ли использован Skill
     */


    const existingSkill =

        analyzeExistingSkill(
            trace
        );



    if (
        existingSkill
    ) {


        return {


            reusable:

                true,


            ...existingSkill


        };


    }









    /*
     * Потом ищем новый Skill
     */


    const newSkill =

        analyzeNewSkill(
            trace
        );



    if (
        newSkill
    ) {


        return {


            reusable:

                true,


            ...newSkill


        };


    }









    /*
     * Нечему учиться
     */


    return {


        action:

            "IGNORE",



        reusable:

            false,



        reason:

            "Повторяемый опыт не найден",



        skillCandidate:

            null


    };


}
