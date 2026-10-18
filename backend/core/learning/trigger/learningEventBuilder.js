/*
 * =========================================================
 * JESSICA LEARNING EVENT BUILDER
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


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


function normalizePayload({

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


function resolveSkillId({

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


function resolveConfidence({

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


function compactAnalysis(
    analysis
) {

    if(
        !isObject(
            analysis
        )
    ){

        return null;

    }


    return {

        action:

            normalizeText(
                analysis.action
            )

            ||

            null,

        reusable:
            analysis.reusable === true,

        reason:

            normalizeText(
                analysis.reason
            ),

        analysisType:

            normalizeText(
                analysis.analysisType
            )

            ||

            null,

        improvementType:

            normalizeText(
                analysis.improvementType
            )

            ||

            null,

        discoveryRequired:
            analysis.discoveryRequired === true,

        metrics:

            isObject(
                analysis.metrics
            )

                ? {
                    ...analysis.metrics
                }

                : {},

        metadata:

            isObject(
                analysis.metadata
            )

                ? {
                    ...analysis.metadata
                }

                : {}

    };

}


function compactDecision(
    decision
) {

    if(
        !isObject(
            decision
        )
    ){

        return null;

    }


    return {

        action:

            normalizeText(
                decision.action
            )

            ||

            null,

        confidence:

            normalizeNumber(
                decision.confidence
            ),

        reusable:
            decision.reusable === true,

        reason:

            normalizeText(
                decision.reason
            ),

        createdAt:

            decision.createdAt

            ||

            null

    };

}


export function buildLearningEvent({

    trace,

    analysis,

    decision

}) {

    const payload =

        normalizePayload({

            decision,

            analysis

        });


    return {

        id:

            trace?.id

            ||

            null,

        traceId:

            trace?.id

            ||

            trace?.traceId

            ||

            null,

        action:

            decision?.action

            ||

            analysis?.action

            ||

            "IGNORE",

        skillId:

            resolveSkillId({

                decision,

                payload

            }),

        confidence:

            resolveConfidence({

                decision,

                analysis,

                payload

            }),

        reusable:

            decision?.reusable === true

            ||

            analysis?.reusable === true,

        reason:

            normalizeText(
                decision?.reason
            )

            ||

            normalizeText(
                analysis?.reason
            )

            ||

            "",

        payload,

        analysis:

            compactAnalysis(
                analysis
            ),

        decision:

            compactDecision(
                decision
            ),

        source:
            "learning-trigger",

        createdAt:

            new Date()
                .toISOString()

    };

}
