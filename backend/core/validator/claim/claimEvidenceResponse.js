import { resolveClaimSourceAlias } from "./claimSourceAlias.js";
import { matchesAttributedEvidence } from "./claimEvidenceMatcher.js";
/*
 * =========================================================
 * JESSICA CLAIM EVIDENCE RESPONSE v1
 * =========================================================
 *
 * Разбор результата Claim Evidence AI.
 *
 *
 * Отвечает за:
 *
 * - очистку JSON;
 * - нормализацию Claims;
 * - проверку Evidence quotation
 *   по реально переданному Source Context.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_EVIDENCE_LENGTH =
    500;


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =

        String(
            text || ""
        )
            .replace(
                /```json/gi,
                ""
            )
            .replace(
                /```/g,
                ""
            )
            .trim();


    const firstBrace =

        value.indexOf(
            "{"
        );


    const lastBrace =

        value.lastIndexOf(
            "}"
        );


    if(
        firstBrace !== -1
        &&
        lastBrace > firstBrace
    ){

        value =

            value.slice(

                firstBrace,

                lastBrace + 1

            );

    }


    return value.trim();

}


/*
 * =========================================================
 * NORMALIZED MATCH TEXT
 * =========================================================
 */


function normalizeMatchText(
    value
) {

    return String(
        value || ""
    )
        .toLowerCase()
        .replace(
            /\u00a0/g,
            " "
        )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}


/*
 * =========================================================
 * NORMALIZE CLAIM
 * =========================================================
 */


function normalizeClaim(
    claim
) {

    if(
        !claim
        ||
        typeof claim !== "object"
        ||
        Array.isArray(claim)
    ){

        return null;

    }


    const text =

        typeof claim.claim === "string"

            ? claim.claim.trim()

            : "";


    if(
        !text
    ){

        return null;

    }


    return {

        claim:
            text,


        supported:

            claim.supported === true,


        evidence:

            typeof claim.evidence === "string"

                ? claim.evidence
                    .trim()
                    .slice(
                        0,
                        MAX_EVIDENCE_LENGTH
                    )

                : "",


        sourceUrl:

            typeof claim.sourceUrl === "string"

                ? claim.sourceUrl.trim()

                : "",


        reason:

            typeof claim.reason === "string"

                ? claim.reason.trim()

                : ""

    };

}


/*
 * =========================================================
 * VERIFY EVIDENCE
 * =========================================================
 */


function verifyEvidenceText(
    claim,
    sourceText,
    originalSources=""
) {

    if(
        claim.supported !== true
    ){

        return claim;

    }


    if(
        !claim.evidence
    ){

        return {

            ...claim,

            supported:
                false,

            reason:

                claim.reason

                ||

                "Не указан подтверждающий фрагмент"

        };

    }


    if (!matchesAttributedEvidence(claim, sourceText)
        && !matchesAttributedEvidence(claim, originalSources)) {
        return {
            ...claim,
            supported: false,
            reason: "Цитата не найдена в указанном первоисточнике"
        };
    }

    return claim;

}


/*
 * =========================================================
 * PARSE
 * =========================================================
 */


export function parseClaimEvidenceResponse(

    raw,

    sourceText,

    originalSources=""

) {

    if(
        !raw
    ){

        return {

            success:
                false,

            claims:
                [],

            error:
                "Validator вернул пустой ответ"

        };

    }


    let parsed;


    try {


        parsed =

            JSON.parse(

                cleanJsonText(
                    raw
                )

            );


    }catch(error){


        return {

            success:
                false,

            claims:
                [],

            error:
                "Validator вернул некорректный JSON"

        };

    }


    const claims =

        Array.isArray(
            parsed?.claims
        )

            ? parsed.claims

                .map(
                    normalizeClaim
                )

                .filter(
                    Boolean
                )

                .map(
                    claim => {
                        const attributed={
                            ...claim,
                            sourceUrl:resolveClaimSourceAlias(claim.sourceUrl,sourceText)
                        };
                        return verifyEvidenceText(attributed,sourceText,originalSources);
                    }
                )

            : [];


    return {

        success:
            true,

        claims,

        error:
            null

    };

}
