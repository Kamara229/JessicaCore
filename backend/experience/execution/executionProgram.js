/*
 * =========================================================
 * JESSICA EXPERIENCE EXECUTION PROGRAM v1
 * =========================================================
 *
 * Безопасный machine contract для Experience Skill.
 *
 *
 * Skill knowledge:
 *
 * workflow / validationRules / strategy
 *      ↓
 * executionProgram
 *      ↓
 * Native Skill Planner
 *
 *
 * ВАЖНО:
 *
 * executionProgram НЕ является произвольным кодом.
 *
 * Поддерживаются только явно разрешённые
 * версии / режимы / стратегии.
 *
 * Это позволяет Jessica выполнять зрелый Skill
 * без AI Planner, сохраняя Groq для задач,
 * где действительно нужна семантическая работа.
 *
 * =========================================================
 */


const PROGRAM_VERSION =
    1;


const MODE_NATIVE =
    "native";


export const EXECUTION_STRATEGY = {

    OFFICIAL_SOURCE_LOOKUP:
        "official_source_lookup"

};


const ALLOWED_STRATEGIES =
    new Set(
        Object.values(
            EXECUTION_STRATEGY
        )
    );


const LEGACY_STRATEGIES =
    new Map([

        [
            "official-source-verification",
            EXECUTION_STRATEGY.OFFICIAL_SOURCE_LOOKUP
        ],

        [
            "official-website-verification",
            EXECUTION_STRATEGY.OFFICIAL_SOURCE_LOOKUP
        ]

    ]);


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


function normalizeStrategy(
    value
) {

    const strategy =

        normalizeText(
            value
        )
            .toLowerCase();


    return ALLOWED_STRATEGIES.has(
        strategy
    )

        ? strategy

        : null;

}


function normalizeEvidenceMode(
    value
) {

    const mode =

        normalizeText(
            value
        )
            .toLowerCase();


    if(
        mode === "source_content"
        ||
        mode === "search_results"
        ||
        mode === "none"
    ){

        return mode;

    }


    return null;

}


/*
 * =========================================================
 * NORMALIZE PROGRAM
 * =========================================================
 */


export function normalizeExecutionProgram(
    value
) {

    if(
        !isObject(
            value
        )
    ){

        return null;

    }


    const version =

        Number(
            value.version || 0
        );


    if(
        version !== PROGRAM_VERSION
    ){

        return null;

    }


    const mode =

        normalizeText(
            value.mode
        )
            .toLowerCase();


    if(
        mode !== MODE_NATIVE
    ){

        return null;

    }


    const strategy =

        normalizeStrategy(
            value.strategy
        );


    if(!strategy){

        return null;

    }


    const evidenceMode =

        normalizeEvidenceMode(
            value.evidenceMode
        )

        ||

        (
            strategy ===
            EXECUTION_STRATEGY.OFFICIAL_SOURCE_LOOKUP

                ? "source_content"

                : "none"
        );


    return {

        version:
            PROGRAM_VERSION,

        mode:
            MODE_NATIVE,

        strategy,

        evidenceMode

    };

}


/*
 * =========================================================
 * LEGACY MIGRATION
 * =========================================================
 *
 * Старые READY Skills появились до executionProgram.
 *
 * Здесь только узкая migration compatibility.
 * После следующего успешного Improvement Program
 * попадёт в новую версию Skill через Learning.
 * =========================================================
 */


function inferLegacyProgram(
    skill
) {

    const id =

        normalizeText(

            skill?.id

            ||

            skill?.skillId

        );


    const strategy =

        LEGACY_STRATEGIES.get(
            id
        );


    if(!strategy){

        return null;

    }


    return normalizeExecutionProgram({

        version:
            PROGRAM_VERSION,

        mode:
            MODE_NATIVE,

        strategy,

        evidenceMode:
            "source_content"

    });

}


/*
 * =========================================================
 * RESOLVE SKILL PROGRAM
 * =========================================================
 */


export function resolveSkillExecutionProgram(
    skill
) {

    return (

        normalizeExecutionProgram(
            skill?.executionProgram
        )

        ||

        inferLegacyProgram(
            skill
        )

    );

}


/*
 * =========================================================
 * SAFE PATTERN INFERENCE
 * =========================================================
 *
 * Dynamic Pattern может получить executable contract
 * только для уже поддерживаемой стратегии и только
 * когда реальный Execution использовал нужный Tool route.
 *
 * Сам AI Pattern Extractor не выбирает исполняемый код.
 * =========================================================
 */


export function inferExecutionProgramFromPattern({

    pattern,

    evidence

} = {}) {

    if(
        !isObject(
            pattern
        )
    ){

        return null;

    }


    const tools =

        new Set(

            Array.isArray(
                evidence?.tools
            )

                ? evidence.tools
                    .map(
                        item =>
                            normalizeText(
                                item
                            )
                    )
                    .filter(Boolean)

                : []

        );


    if(
        !tools.has(
            "web_search"
        )

        ||

        !tools.has(
            "web_fetch"
        )
    ){

        return null;

    }


    const semanticText =

        [

            pattern.name,

            pattern.description,

            ...(
                Array.isArray(
                    pattern.triggerPatterns
                )

                    ? pattern.triggerPatterns

                    : []
            ),

            ...(
                Array.isArray(
                    pattern.validationRules
                )

                    ? pattern.validationRules

                    : []
            )

        ]
            .map(
                normalizeText
            )
            .filter(Boolean)
            .join(
                " "
            )
            .toLowerCase();


    const officialSource =

        /официальн\p{L}*/iu.test(
            semanticText
        )

        ||

        /\bofficial\b/iu.test(
            semanticText
        );


    const sourceMeaning =

        /(источник|сайт|страниц|source|website|site|page)/iu
            .test(
                semanticText
            );


    if(
        !officialSource
        ||
        !sourceMeaning
    ){

        return null;

    }


    return normalizeExecutionProgram({

        version:
            PROGRAM_VERSION,

        mode:
            MODE_NATIVE,

        strategy:
            EXECUTION_STRATEGY.OFFICIAL_SOURCE_LOOKUP,

        evidenceMode:
            "source_content"

    });

}
