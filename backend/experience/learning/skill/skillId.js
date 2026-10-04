import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA EXPERIENCE SKILL ID
 * =========================================================
 *
 * Формирование стабильного Skill ID.
 *
 * =========================================================
 */


import {
    normalizeText
} from "./skillUtils.js";


const TRANSLITERATION = {

    а:"a",
    б:"b",
    в:"v",
    г:"g",
    д:"d",
    е:"e",
    ё:"e",
    ж:"zh",
    з:"z",
    и:"i",
    й:"y",
    к:"k",
    л:"l",
    м:"m",
    н:"n",
    о:"o",
    п:"p",
    р:"r",
    с:"s",
    т:"t",
    у:"u",
    ф:"f",
    х:"h",
    ц:"ts",
    ч:"ch",
    ш:"sh",
    щ:"sch",
    ъ:"",
    ы:"y",
    ь:"",
    э:"e",
    ю:"yu",
    я:"ya"

};


/*
 * =========================================================
 * BUILD SKILL ID
 * =========================================================
 */


export function buildLearningSkillId(
    value
) {

    const normalized =

        normalizeText(
            value
        )

        .toLowerCase()

        .split("")

        .map(

            char =>

                TRANSLITERATION[char]

                ??

                char

        )

        .join("")

        .replace(
            /[^a-z0-9]+/g,
            "-"
        )

        .replace(
            /^-+|-+$/g,
            ""
        )

        .slice(
            0,
            80
        );


    if(
        normalized
    ){

        return normalized;

    }


    return (

        "skill-"

        +

        randomUUID()
            .slice(
                0,
                8
            )

    );

}
