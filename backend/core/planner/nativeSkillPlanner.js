/*
 * =========================================================
 * JESSICA NATIVE SKILL PLANNER v4
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


// Ensure the direct native plan has its web_fetch executor registered.
import "../../tools/webFetchTool.js";

import {
    validatePlan
} from "./planValidator.js";


import {
    extractNativeDates
} from "../native/dateUtils.js";
import { extractExplicitDateRange } from "../native/explicitDateRange.js";
import {extractSameDayRange} from "../native/sameDayRange.js";

import { hasVerifiedDateSkillEvidence } from "./youngDateSkill.js";
import { dateDiffMode } from "./dateDiffMode.js";
import { buildNativeDocumentFetchPlan } from "./nativeDocumentFetchPlan.js";
import {buildNativeIntegerMultiplicationPlan} from "./nativeIntegerMultiplicationPlan.js";
import { buildNativeSourceComparisonPlan } from "./nativeSourceComparisonPlan.js";
import { buildNativeSingleLicensePlan } from "./nativeSingleLicensePlan.js";
import { tryCreateKnownLicenseComparison } from "./knownLicenseComparison.js";


import {
    EXECUTION_STRATEGY,
    resolveSkillExecutionProgram
} from "../../experience/execution/executionProgram.js";


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


    const bootstrapExecutable =

        skill?.metadata
            ?.bootstrap
            ?.executable === true;


    if(bootstrapExecutable){

        return true;

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

    dependencyContext,

    executionExperience

}) {

    const taskDates =

        extractNativeDates(
            task
        );


    const dependencyDates =

        extractNativeDates(
            dependencyContext
        );

    const explicitRange = extractExplicitDateRange(task);
    const sameDayRange = extractSameDayRange(task);


    let startDate =
        null;


    let endDate =
        null;


    if(explicitRange){

        startDate = explicitRange.startDate;
        endDate = explicitRange.endDate;

    }
    else if(sameDayRange){
        startDate=sameDayRange.startDate;
        endDate=sameDayRange.endDate;
    }
    else if(
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
        (startDate === endDate && !sameDayRange)
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

                || taskText.includes("вычисл")

                || taskText.includes("количеств")

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


    const skill =

        findExecutableSkill(

            executionExperience,

            EXECUTION_STRATEGY
                .DATE_DIFFERENCE

        );


    const skillId =

        normalizeSkillId(
            skill
        );


    const elapsedMode = dateDiffMode(taskText) === "elapsed";

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

        experience:

            skillId

                ? {

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
                                    skill?.name
                                )

                        }

                    ]

                }

                : {

                    used:
                        false,

                    source:
                        "native-capability",

                    skills:
                        []

                },

        steps: [

            {

                id:
                    "native_date_diff",

                tool:
                    "date_diff",

                arguments: {

                    startDate,

                    endDate,

                    mode:

                        elapsedMode

                            ? "elapsed"

                            : "difference"

                },

                experienceUsed:

                    Boolean(
                        skillId
                    )

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


    // Neutral English query for an exact Python release index.
    // No guessed version number is ever included.
    if (/\bpython\b/iu.test(text)
        && /(верси|релиз|выпуск|latest|release|version)/iu.test(text)) {
        return "Python latest stable release date official downloads python.org";
    }

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


/*
 * =========================================================
 * RELEASE FACT SIGNAL
 * =========================================================
 */


function shouldExtractReleaseFact(
    task
) {

    const text =

        normalizeText(
            task
        )
            .toLowerCase();


    const versionSignal =

        /(верси|version|release|релиз|выпуск)/iu
            .test(
                text
            );


    const freshnessSignal =

        /(актуальн|текущ|последн|стабильн|latest|current|stable)/iu
            .test(
                text
            );


    return (

        versionSignal

        &&

        freshnessSignal

    );

}


function findExecutableSkill(
    executionExperience,
    strategy
) {

    return getSkills(
        executionExperience
    )
        .find(
            skill => {

                const program =

                    resolveSkillExecutionProgram(
                        skill
                    );


                if (program?.strategy !== strategy) return false;

                // NEW/LEARNING date Skills are already deterministic;
                // allow a trusted one to gain maturity through real executions.
                if (strategy === EXECUTION_STRATEGY.DATE_DIFFERENCE) {
                    return isReadySkill(skill) || hasVerifiedDateSkillEvidence(skill);
                }

                return isReadySkill(skill);

            }
        )

        ||

        null;

}


function buildOfficialSourcePlan({

    task,

    executionExperience

}) {

    const skill =

        findExecutableSkill(

            executionExperience,

            EXECUTION_STRATEGY
                .OFFICIAL_SOURCE_LOOKUP

        );


    if(!skill){

        return null;

    }


    const program =

        resolveSkillExecutionProgram(
            skill
        );


    if(!program){

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
                program.evidenceMode,

            reason:
                "Executable Experience требует проверки факта по содержимому официального источника."

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

            },

            ...(
                shouldExtractReleaseFact(
                    task
                )

                    ? [

                        {

                            id:
                                "native_release_fact",

                            tool:
                                "release_fact",

                            arguments: {

                                content: {

                                    $from:
                                        "native_fetch_source",

                                    path:
                                        "data.content"

                                },

                                sourceUrl: {

                                    $from:
                                        "native_fetch_source",

                                    path:
                                        "data.url"

                                },

                                sourceTitle: {

                                    $from:
                                        "native_fetch_source",

                                    path:
                                        "data.title"

                                },

                                subject:
                                    task

                            },

                            experienceUsed:
                                true

                        }

                    ]

                    : []
            )

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


    // GPLv3 is the canonical GPL 3.0 name, not an invented version.
    const groundedTask=source==="native-single-license-source" &&
        plan?.metadata?.licenseId==="GPL-3.0" && /\bgpl\s*v\s*3\b/iu.test(task)
        ? task+" GPL 3.0" : task;
    const validation =

        validatePlan(

            plan,

            context,

            groundedTask

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


    // Two canonical license texts outrank any generic single-source Skill.
    const licensePair = tryCreateKnownLicenseComparison({task, planningContext});
    if (licensePair?.handled === true) return licensePair;

    const paired = buildNativeSourceComparisonPlan(task);
    if (paired) {
        return finalizeNativePlan({
            task, context: planningContext, plan: paired,
            source: "native-two-source-comparison"
        });
    }

    // An explicitly named primary-text license source precedes generic
    // search-based Skills. Retrieval does not imply any Skill was applied.
    const licensePlan = buildNativeSingleLicensePlan(task);
    if (licensePlan) {
        return finalizeNativePlan({
            task, context: planningContext, plan: licensePlan,
            source: "native-single-license-source"
        });
    }

    const documentPlan = buildNativeDocumentFetchPlan(task);
    if (documentPlan) {
        return finalizeNativePlan({
            task,
            context: planningContext,
            plan: documentPlan,
            source: "native-document-capture"
        });
    }

    const productPlan=buildNativeIntegerMultiplicationPlan(task,executionExperience);
    if(productPlan) return finalizeNativePlan({task,context:planningContext,plan:productPlan,source:productPlan.experience.used?"experience:integer-multiplication":"native-integer-multiplication"});
    const datePlan =

        buildDateDiffPlan({

            task,

            dependencyContext,

            executionExperience

        });


    if(datePlan){

        return finalizeNativePlan({

            task,

            context:
                planningContext,

            plan:
                datePlan,

            source:

                datePlan
                    ?.experience
                    ?.used === true

                    ? (
                        "experience:"
                        +
                        normalizeText(
                            datePlan
                                ?.experience
                                ?.skills
                                ?.[0]
                                ?.id
                        )
                    )

                    : "native-date-diff"

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

                (
                    "experience:"
                    +
                    normalizeText(

                        officialSourcePlan
                            ?.experience
                            ?.skills
                            ?.[0]
                            ?.id

                    )
                )

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
