/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER v2
 * =========================================================
 *
 * Анализирует Execution Trace.
 *
 *
 * Главная задача:
 *
 * определить:
 *
 * - можно ли превратить опыт в Skill;
 * - создать новый Skill;
 * - улучшить существующий Skill;
 * - игнорировать опыт.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - пишет в Supabase;
 * - изменяет Experience.
 *
 * Только создаёт Learning Candidate.
 *
 * =========================================================
 */



/*
 * =========================================================
 * REUSABLE EXPERIENCE PATTERNS
 * =========================================================
 *
 * Временная база паттернов.
 *
 * Позже заменится на:
 *
 * Experience Memory
 *        +
 * AI extraction
 *
 * =========================================================
 */


const REUSABLE_PATTERNS = [

    {

        id:
            "official-source-verification",


        name:
            "Проверка официального источника",


        category:
            "verification",


        description:
            "Навык поиска, проверки и подтверждения официальных источников информации.",


        keywords:
        [

            "официальный сайт",

            "официальный источник",

            "документ",

            "сертификат",

            "проверить",

            "подтверждение"

        ],


        workflow:
        [

            "найти потенциальный источник",

            "проверить принадлежность источника",

            "сопоставить данные",

            "сформировать подтверждённый вывод"

        ],


        validationRules:
        [

            "использовать первичные источники",

            "проверять соответствие источника запросу",

            "не использовать неподтверждённые данные"

        ]

    },




    {

        id:
            "company-contact-finder",


        name:
            "Поиск контактов организации",


        category:
            "web-research",


        description:
            "Навык поиска контактной информации организаций.",


        keywords:
        [

            "контакт",

            "почта",

            "email",

            "телефон",

            "адрес компании",

            "support"

        ],


        workflow:
        [

            "найти официальный ресурс",

            "извлечь контактные данные",

            "проверить актуальность",

            "предоставить результат"

        ],


        validationRules:
        [

            "использовать официальный источник",

            "проверять актуальность контактов"

        ]

    }

];







/*
 * =========================================================
 * HELPERS
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






function countExamples(
    trace
) {

    if (
        Array.isArray(
            trace?.examples
        )
    ) {

        return trace.examples.length;

    }


    return 1;

}







/*
 * =========================================================
 * FIND PATTERN
 * =========================================================
 */


function detectReusablePattern(
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





    for(
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



        if(
            matches > score
        ){

            score =
                matches;


            best =
                pattern;

        }

    }




    if(
        !best
    ){

        return null;

    }





    return {


        ...best,


        confidence:

            Math.min(

                score /
                best.keywords.length,

                1

            )


    };


}








/*
 * =========================================================
 * EXISTING SKILL IMPROVEMENT
 * =========================================================
 */


function analyzeExistingSkill(
    trace
) {


    const usage =
        trace?.experienceUsage;



    if(
        !usage ||
        usage.used !== true
    ){

        return null;

    }



    if(
        !Array.isArray(
            usage.skills
        )
        ||
        usage.skills.length === 0
    ){

        return null;

    }





    return {


        action:

            "SKILL_IMPROVEMENT",



        reusable:

            true,



        reason:

            "Существующий Skill получил новый успешный опыт",



        skillCandidate:

        {

            skills:

                usage.skills,



            examples:

            [

                {

                    task:

                        trace.task,


                    result:

                        trace.result || "",


                    success:

                        true

                }

            ],



            confidence:

                0.9,


            source:

                "execution-trace"

        }


    };

}









/*
 * =========================================================
 * NEW SKILL
 * =========================================================
 */


function analyzeNewSkill(
    trace
) {


    const pattern =

        detectReusablePattern(
            trace.task
        );



    if(
        !pattern
    ){

        return null;

    }






    return {


        action:

            "NEW_SKILL",



        reusable:

            true,



        reason:

            "Обнаружен повторяемый сценарий, пригодный для Skill",



        skillCandidate:

        {


            skillId:

                pattern.id,


            name:

                pattern.name,


            category:

                pattern.category,


            description:

                pattern.description,



            workflow:

                pattern.workflow,



            validationRules:

                pattern.validationRules,



            examples:

            [

                {

                    task:

                        trace.task,


                    result:

                        trace.result || "",


                    success:

                        true

                }

            ],



            constraints:

            [],



            confidence:

                pattern.confidence,



            source:

                "execution-trace"

        }


    };

}









/*
 * =========================================================
 * MAIN
 * =========================================================
 */


export function analyzeExecutionTrace(
    trace
) {


    if(
        !trace ||
        typeof trace !== "object"
    ){

        return {

            action:
                "IGNORE",

            reusable:
                false,

            reason:
                "Execution Trace отсутствует",

            skillCandidate:
                null

        };

    }





    if(
        (trace.stats?.completed || 0) <= 0
    ){

        return {

            action:
                "IGNORE",

            reusable:
                false,

            reason:
                "Нет успешного выполнения",

            skillCandidate:
                null

        };

    }








    /*
     * 1. Улучшение существующего Skill
     */


    const existingSkill =

        analyzeExistingSkill(
            trace
        );



    if(
        existingSkill
    ){

        return existingSkill;

    }








    /*
     * 2. Новый Skill
     */


    const newSkill =

        analyzeNewSkill(
            trace
        );



    if(
        newSkill
    ){

        return newSkill;

    }








    /*
     * 3. Игнор
     */


    return {

        action:

            "IGNORE",


        reusable:

            false,


        reason:

            "Недостаточно признаков повторяемого навыка",


        skillCandidate:

            null

    };

}
