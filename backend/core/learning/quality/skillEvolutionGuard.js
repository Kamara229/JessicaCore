/*
 * =========================================================
 * JESSICA SKILL EVOLUTION GUARD v1
 * =========================================================
 *
 * Решает, нужна ли новая версия Skill.
 *
 * NEW / LEARNING:
 * metrics progression сохраняется версией,
 * чтобы Skill мог дойти до READY.
 *
 * READY:
 * чистое подтверждение существующего знания
 * сохраняется только как Evidence — без v17, v18, v100...
 * =========================================================
 */


export const EVOLUTION_MODE = {

    NEW_VERSION:
        "NEW_VERSION",

    REINFORCEMENT_ONLY:
        "REINFORCEMENT_ONLY"

};


function getEvolution(
    proposal
) {

    const value =

        proposal
            ?.analysis
            ?.evolution;


    return (

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    )

        ? value

        : null;

}


export function evaluateSkillEvolution(
    proposal
) {

    if(
        proposal?.action !==
        "SKILL_IMPROVEMENT"
    ){

        return {

            mode:
                EVOLUTION_MODE.NEW_VERSION,

            reason:
                "Новый Skill требует собственной версии"

        };

    }


    const evolution =

        getEvolution(
            proposal
        );


    /*
     * Старые Queue Items без evolution metadata
     * сохраняем по прежнему безопасному поведению.
     */


    if(!evolution){

        return {

            mode:
                EVOLUTION_MODE.NEW_VERSION,

            reason:
                "Evolution metadata отсутствует; используется совместимый versioned flow",

            legacy:
                true

        };

    }


    if(
        evolution.knowledgeChanged === true
    ){

        return {

            mode:
                EVOLUTION_MODE.NEW_VERSION,

            reason:
                "Изменилось содержательное знание Skill",

            evolution

        };

    }


    if(
        evolution.maturityChanged === true
    ){

        return {

            mode:
                EVOLUTION_MODE.NEW_VERSION,

            reason:
                "Skill перешёл на новый уровень зрелости",

            evolution

        };

    }


    const baseMaturity =

        String(
            evolution.baseMaturityLevel || ""
        )
            .trim()
            .toUpperCase();


    /*
     * До READY сохраняем metric progression.
     * Количество таких версий ограничено процессом зрелости.
     */


    if(
        baseMaturity !== "READY"
    ){

        return {

            mode:
                EVOLUTION_MODE.NEW_VERSION,

            reason:
                "Skill ещё не READY; сохраняется progression Learning Metrics",

            evolution

        };

    }


    return {

        mode:
            EVOLUTION_MODE.REINFORCEMENT_ONLY,

        reason:
            "READY Skill получил только новое подтверждение существующего знания",

        evolution

    };

}
