/*
 * =========================================================
 * JESSICA SKILL IMPROVEMENT REFERENCE
 * =========================================================
 */


import {
    normalizeText
} from "../candidateUtils.js";


export function buildSkillReference(
    rawSkill,
    resolvedSkill
) {

    const id =

        normalizeText(

            resolvedSkill?.id

            ||

            resolvedSkill?.skillId

            ||

            rawSkill?.id

            ||

            rawSkill?.skillId

        );


    if(!id){

        return null;

    }


    const version =

        Number(

            resolvedSkill?.version

            ??

            rawSkill?.version

            ??

            0

        );


    return {

        id,

        version:

            Number.isFinite(version)
            &&
            version > 0

                ? version

                : null

    };

}
