/*
 * =========================================================
 * JESSICA LEARNING EVENT PAYLOAD
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


export function normalizeLearningEventPayload({

    decision,

    analysis

}) {

    const routerPayload =

        isObject(
            decision?.payload
        )

            ? decision.payload

            : {};


    const candidate =

        isObject(
            analysis?.skillCandidate
        )

            ? analysis.skillCandidate

            : null;


    const payload = {
        ...routerPayload
    };


    if(
        !isObject(
            payload.skillCandidate
        )
        &&
        candidate
    ){

        payload.skillCandidate =
            candidate;

    }


    if(
        !Array.isArray(
            payload.skills
        )
        &&
        Array.isArray(
            candidate?.skills
        )
    ){

        payload.skills =
            candidate.skills;

    }


    if(
        !normalizeText(
            payload.source
        )
    ){

        payload.source =
            "experience-analyzer";

    }


    return payload;

}


export function resolveLearningEventSkillId({

    decision,

    payload

}) {

    const candidate =

        isObject(
            payload?.skillCandidate
        )

            ? payload.skillCandidate

            : null;


    const candidateSkillId =

        normalizeText(

            candidate?.targetSkillId

            ||

            candidate?.skillId

            ||

            candidate?.id

        );


    if(candidateSkillId){

        return candidateSkillId;

    }


    const skills =

        Array.isArray(
            payload?.skills
        )

            ? payload.skills

            : [];


    if(
        skills.length > 0
    ){

        const existingSkillId =

            normalizeText(

                skills[0]?.id

                ||

                skills[0]?.skillId

            );


        if(existingSkillId){

            return existingSkillId;

        }

    }


    return (

        normalizeText(
            decision?.skillId
        )

        ||

        null

    );

}


export function resolveLearningEventConfidence({

    decision,

    analysis,

    payload

}) {

    const values = [

        decision?.confidence,

        payload
            ?.skillCandidate
            ?.confidence,

        analysis
            ?.skillCandidate
            ?.confidence,

        analysis
            ?.metrics
            ?.confidence

    ];


    for(
        const value
        of values
    ){

        const number =
            Number(value);


        if(
            Number.isFinite(number)
        ){

            return Math.max(

                0,

                Math.min(
                    1,
                    number
                )

            );

        }

    }


    return 0;

}
