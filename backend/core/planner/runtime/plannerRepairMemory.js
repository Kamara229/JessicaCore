/*
 * =========================================================
 * JESSICA PLANNER REPAIR MEMORY v1
 * =========================================================
 *
 * Хранит только компактные semantic constraints,
 * которые уже были выявлены машинной validation
 * внутри одного createPlan().
 *
 * Это не history reasoning и не Experience.
 *
 * Нужно, чтобы новая semantic ошибка
 * не стирала ограничения предыдущей попытки.
 *
 * =========================================================
 */


function normalize(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toLowerCase();

}


export function createPlannerRepairMemory()
{

    return {

        unknownFactGrounding:
            false,

        sourceContentRequired:
            false

    };

}


export function rememberPlannerRejection(
    memory,
    reason
) {

    const target =
        memory
        &&
        typeof memory === "object"
            ? memory
            : createPlannerRepairMemory();


    const text =
        normalize(
            reason
        );


    if(
        text.includes(
            "неподтверждённое конкретное значение"
        )
        ||
        text.includes(
            "неподтвержденное конкретное значение"
        )
        ||
        (
            text.includes(
                "отсутствует в исходной задаче"
            )
            &&
            text.includes(
                "подтвержд"
            )
        )
    ){

        target.unknownFactGrounding =
            true;

    }


    if(
        (
            text.includes(
                "evidence.mode=source_content"
            )
            &&
            (
                text.includes(
                    "требуется"
                )
                ||
                text.includes(
                    "обязательно"
                )
            )
        )
        ||
        (
            text.includes(
                "недостаточный evidence.mode"
            )
            &&
            text.includes(
                "source_content"
            )
        )
    ){

        target.sourceContentRequired =
            true;

    }


    return target;

}


export function buildPlannerRepairMemoryText(
    memory
) {

    if(
        !memory
        ||
        typeof memory !== "object"
    ){

        return "";

    }


    const lines =
        [];


    if(
        memory.unknownFactGrounding === true
    ){

        lines.push(
            "- Искомое конкретное значение всё ещё UNKNOWN: не подставляй никакую предполагаемую версию, дату, номер релиза, ID или иной ответ в search query, URL или tool arguments. Ищи сам факт нейтральной формулировкой."
        );

    }


    if(
        memory.sourceContentRequired === true
    ){

        lines.push(
            "- Для этой задачи evidence.mode обязан оставаться source_content; план должен содержать web_fetch. Не понижай evidence до search_results."
        );

    }


    if(
        lines.length === 0
    ){

        return "";

    }


    return [

        "=== НАКОПЛЕННЫЕ МАШИННЫЕ ОГРАНИЧЕНИЯ ===",

        ...lines,

        "Эти ограничения действуют одновременно и не отменяются более поздней ошибкой Validator."

    ]
        .join(
            "\n"
        );

}
