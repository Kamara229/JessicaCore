/*
 * =========================================================
 * JESSICA NATIVE CAPABILITY PLANNER v1
 * =========================================================
 *
 * Best-effort deterministic planning,
 * когда AI Planner недоступен.
 *
 *
 * Это fallback, а НЕ замена Experience.
 *
 * Используется только после неудачи AI Planner,
 * если task можно безопасно сопоставить
 * с уже существующей native capability.
 *
 * =========================================================
 */


import {
    validatePlan
} from "./planValidator.js";
import { sanitizeSourceUrlInput } from "../../tools/sourceUrlInput.js";


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}


function extractFirstUrl(
    task
) {

    const match =

        normalizeText(
            task
        )
            .match(
                /https?:\/\/[^\s<>"')\]]+/i
            );


    return match?.[0]
        ? sanitizeSourceUrlInput(match[0])
        : "";

}


function asksForExternalLookup(
    task
) {

    const text =
        normalizeText(
            task
        );


    return (

        /(найд\p{L}*|поищ\p{L}*|проверь\p{L}*|узнай\p{L}*|актуальн\p{L}*|текущ\p{L}*|последн\p{L}*|официальн\p{L}*)/iu
            .test(
                text
            )

        ||

        /\b(find|search|look\s+up|verify|check|latest|current|official)\b/iu
            .test(
                text
            )

    );

}


function requiresSourceContent(
    task
) {

    const text =
        normalizeText(
            task
        );


    return (

        /официальн\p{L}*/iu.test(
            text
        )

        ||

        /(открой\p{L}*|прочитай\p{L}*|проверь\p{L}*\s+содержим\p{L}*|по\s+содержим\p{L}*)/iu
            .test(
                text
            )

        ||

        /\b(official|open|read|page content|source content)\b/iu
            .test(
                text
            )

    );

}


function finalize({

    task,

    context,

    plan,

    source

}) {

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

            "Jessica Native Capability Plan rejected:",

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
 * DIRECT URL
 * =========================================================
 */


function buildDirectUrlPlan(
    task
) {

    const url =

        extractFirstUrl(
            task
        );


    if(!url){

        return null;

    }


    return {

        intent:
            "native_direct_source",

        requiresTools:
            true,

        reasoningSummary:
            "В задаче уже указан URL; Jessica читает источник напрямую без AI Planner.",

        evidence: {

            mode:
                "source_content",

            reason:
                "Пользователь указал конкретный внешний источник."

        },

        steps: [

            {

                id:
                    "native_fetch_direct",

                tool:
                    "web_fetch",

                arguments: {
                    url
                }

            }

        ],

        metadata: {

            plannerVersion:
                "native-capability-v1",

            generatedAt:
                new Date()
                    .toISOString()

        }

    };

}


/*
 * =========================================================
 * GENERIC LOOKUP
 * =========================================================
 */


function buildLookupPlan(
    task
) {

    if(
        !asksForExternalLookup(
            task
        )
    ){

        return null;

    }


    const query =

        normalizeText(
            task
        )
            .slice(
                0,
                900
            );


    if(!query){

        return null;

    }


    const sourceContent =

        requiresSourceContent(
            task
        );


    const steps = [

        {

            id:
                "native_search",

            tool:
                "web_search",

            arguments: {
                query
            }

        }

    ];


    if(sourceContent){

        steps.push({

            id:
                "native_fetch",

            tool:
                "web_fetch",

            arguments: {

                url: {

                    $from:
                        "native_search",

                    path:
                        "data.results.0.url"

                }

            }

        });

    }


    return {

        intent:
            "native_external_lookup",

        requiresTools:
            true,

        reasoningSummary:
            "AI Planner недоступен; используется безопасный встроенный маршрут поиска.",

        evidence: {

            mode:

                sourceContent

                    ? "source_content"

                    : "search_results",

            reason:

                sourceContent

                    ? "Нужно проверить содержимое найденного источника."

                    : "Для best-effort ответа используется поисковая выдача."

        },

        steps,

        metadata: {

            plannerVersion:
                "native-capability-v1",

            generatedAt:
                new Date()
                    .toISOString()

        }

    };

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function tryCreateNativeCapabilityPlan({

    task,

    planningContext = {}

} = {}) {

    const direct =

        buildDirectUrlPlan(
            task
        );


    if(direct){

        return finalize({

            task,

            context:
                planningContext,

            plan:
                direct,

            source:
                "native-direct-url"

        });

    }


    const lookup =

        buildLookupPlan(
            task
        );


    if(lookup){

        return finalize({

            task,

            context:
                planningContext,

            plan:
                lookup,

            source:
                "native-lookup-fallback"

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
