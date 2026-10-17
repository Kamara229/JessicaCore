/*
 * =========================================================
 * JESSICA NATIVE SKILL PLANNER v1
 * =========================================================
 *
 * Skill-first planning без AI.
 *
 *
 * Используется только для маршрутов,
 * которые Jessica уже умеет выполнять
 * детерминированно.
 *
 *
 * Priority:
 *
 * native capability
 *      ↓
 * READY Experience Skill
 *      ↓
 * AI Planner fallback
 *
 *
 * КРИТИЧЕСКО:
 *
 * Этот модуль не пытается заменить
 * универсальный Planner.
 *
 * Он забирает только те задачи,
 * для которых существует безопасный
 * machine route.
 *
 * =========================================================
 */


import {
    validatePlan
} from "./planValidator.js";


import {
    extractNativeDates
} from "../native/dateUtils.js";


const NATIVE_OFFICIAL_SOURCE_SKILLS =
    new Set([

        "official-source-verification",

        "official-website-verification"

    ]);


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
        .trim();

}


function normalizeUnit(
    value
) {

    const number =
        Number(value);


    if(
        !Number.isFinite(
            number
        )
    ){

        return 0;

    }


    return Math.max(
        0,
        Math.min(
            1,
            number
        )
    );

}


function normalizeSkillId(
    skill
) {

    return normalizeText(

        skill?.id

        ||

        skill?.skillId

    );

}


function getSkills(
    executionExperience
) {

    return Array.isArray(
        executionExperience?.skills
    )

        ? executionExperience.skills.filter(
            item =>
                item
                &&
                typeof item === "object"
                &&
                !Array.isArray(item)
        )

        : [];

}


/*
 * =========================================================
 * READY SKILL
 * =========================================================
 */


function isReadySkill(
    skill
) {

    if(
        !skill
        ||
        typeof skill !== "object"
        ||
        Array.isArray(skill)
        ||
        skill.enabled === false
    ){

        return false;

    }


    const maturityLevel =

        normalizeText(
            skill?.learning?.maturityLevel
        )
            .toUpperCase();


    if(
        maturityLevel === "READY"
    ){

        return true;

    }


    const occurrences =

        Number(
            skill?.learning?.occurrences || 0
        );


    const confidence =

        normalizeUnit(

            skill?.learning?.confidence

            ??

            skill?.confidence

        );


    return (

        occurrences >= 3

        &&

        confidence >= 0.85

    );

}


/*
 * =========================================================
 * DATE DIFF
 * =========================================================
 */


function buildDateDiffPlan({

    task,

    dependencyContext

}) {

    const taskDates =

        extractNativeDates(
            task
        );


    const dependencyDates =

        extractNativeDates(
            dependencyContext
        );


    let startDate =
        null;


    let endDate =
        null;


    if(
        dependencyDates.length > 0
        &&
        taskDates.length > 0
    ){

        startDate =
            dependencyDates[0].iso;


        endDate =
            taskDates[
                taskDates.length - 1
            ].iso;

    }
    else if(
        taskDates.length >= 2
    ){

        startDate =
            taskDates[0].iso;


        endDate =
            taskDates[1].iso;

    }


    if(
        !startDate
        ||
        !endDate
        ||
        startDate === endDate
    ){

        return null;

    }


    const taskText =

        normalizeText(
            task
        )
            .toLowerCase();


    const asksForDays =

        (
            taskText.includes(
                "дн"
            )

            &&

            (
                taskText.includes(
                    "сколько"
                )

                ||

                taskText.includes(
                    "рассч"
                )

                ||

                taskText.includes(
                    "разниц"
                )

                ||

                taskText.includes(
                    "прошло"
                )
            )
        )

        ||

        (
            taskText.includes(
                "day"
            )

            &&

            (
                taskText.includes(
                    "how many"
                )

                ||

                taskText.includes(
                    "difference"
                )

                ||

                taskText.includes(
                    "calculate"
                )
            )
        );


    if(
        !asksForDays
    ){

        return null;

    }


    return {

        intent:
            "native_date_difference",

        requiresTools:
            true,

        reasoningSummary:
            "Даты уже известны; используется детерминированный локальный расчёт без AI.",

        evidence: {

            mode:
                "none",

            reason:
                "Для арифметики по уже полученным датам внешнее подтверждение не требуется."

        },

        steps: [

            {

                id:
                    "native_date_diff",

                tool:
                    "date_diff",

                arguments: {

                    startDate,

                    endDate

                },

                experienceUsed:
                    false

            }

        ],

        metadata: {

            plannerVersion:
                "native-skill-v1",

            generatedAt:
                new Date()
                    .toISOString()

        }

    };

}


/*
 * =========================================================
 * OFFICIAL SOURCE VERIFICATION
 * =========================================================
 */


function buildOfficialSourceSearchQuery(
    task
) {

    const text =
        normalizeText(
            task
        );


    if(
        !text
    ){

        return "";
    }


    /*
     * Не угадываем искомое значение.
     *
     * Сам текст задачи уже содержит
     * предмет поиска и является grounded.
     */


    return text

        .replace(
            /\s+/g,
            " "
        )

        .slice(
            0,
            900
        );

}


function findOfficialSourceSkill(
    executionExperience
) {

    return getSkills(
        executionExperience
    )
        .find(
            skill =>

                NATIVE_OFFICIAL_SOURCE_SKILLS.has(
                    normalizeSkillId(
                        skill
                    )
                )

                &&

                isReadySkill(
                    skill
                )
        )

        ||

        null;

}


function buildOfficialSourcePlan({

    task,

    executionExperience

}) {

    const skill =

        findOfficialSourceSkill(
            executionExperience
        );


    if(!skill){

        return null;

    }


    const query =

        buildOfficialSourceSearchQuery(
            task
        );


    if(!query){

        return null;

    }


    const skillId =

        normalizeSkillId(
            skill
        );


    return {

        intent:
            "native_official_source_verification",

        requiresTools:
            true,

        reasoningSummary:

            (
                "READY Experience Skill задаёт безопасный маршрут: "
                +
                "нейтральный поиск источника и чтение выбранной официальной страницы."
            ),

        evidence: {

            mode:
                "source_content",

            reason:
                "Искомый актуальный факт должен быть проверен по содержимому официального источника."

        },

        experience: {

            used:
                true,

            source:
                "native-skill",

            skills: [

                {

                    id:
                        skillId,

                    name:

                        normalizeText(
                            skill.name
                        )

                }

            ]

        },

        steps: [

            {

                id:
                    "native_search_source",

                tool:
                    "web_search",

                arguments: {

                    query

                },

                experienceUsed:
                    true

            },

            {

                id:
                    "native_fetch_source",

                tool:
                    "web_fetch",

                arguments: {

                    url: {

                        $from:
                            "native_search_source",

                        path:
                            "data.results.0.url"

                    }

                },

                experienceUsed:
                    true

            }

        ],

        metadata: {

            plannerVersion:
                "native-skill-v1",

            generatedAt:
                new Date()
                    .toISOString()

        }

    };

}


/*
 * =========================================================
 * VALIDATE
 * =========================================================
 */


function finalizeNativePlan({

    task,

    context,

    plan,

    source

}) {

    if(!plan){

        return {

            handled:
                false,

            plan:
                null,

            source:
                null

        };

    }


    const validation =

        validatePlan(

            plan,

            context,

            task

        );


    if(
        validation?.success !== true
    ){

        console.warn(

            "Jessica Native Skill Plan rejected:",

            {

                source,

                reason:

                    validation?.text

                    ||

                    "unknown validation error"

            }

        );


        return {

            handled:
                false,

            plan:
                null,

            source:
                null,

            validation

        };

    }


    return {

        handled:
            true,

        plan,

        source,

        validation

    };

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function tryCreateNativeSkillPlan({

    task,

    executionExperience,

    dependencyContext = "",

    planningContext = {}

} = {}) {

    /*
     * Native capability имеет приоритет:
     * AI не нужен даже без Experience.
     */


    const datePlan =

        buildDateDiffPlan({

            task,

            dependencyContext

        });


    if(datePlan){

        return finalizeNativePlan({

            task,

            context:
                planningContext,

            plan:
                datePlan,

            source:
                "native-date-diff"

        });

    }


    /*
     * Затем READY Experience.
     */


    const officialSourcePlan =

        buildOfficialSourcePlan({

            task,

            executionExperience

        });


    if(officialSourcePlan){

        return finalizeNativePlan({

            task,

            context:
                planningContext,

            plan:
                officialSourcePlan,

            source:
                "experience:official-source-verification"

        });

    }


    return {

        handled:
            false,

        plan:
            null,

        source:
            null

    };

}
