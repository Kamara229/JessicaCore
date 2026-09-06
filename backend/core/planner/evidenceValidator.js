/*
 * =========================================================
 * JESSICA EVIDENCE VALIDATOR
 * =========================================================
 *
 * Проверяет, соответствует ли план
 * заявленному уровню доказательств.
 *
 * Поддерживаемые режимы:
 *
 * none
 * search_results
 * source_content
 */


const ALLOWED_EVIDENCE_MODES =
    new Set([
        "none",
        "search_results",
        "source_content"
    ]);


/*
 * =========================================================
 * VALIDATE EVIDENCE STRUCTURE
 * =========================================================
 */


export function validateEvidenceStructure(
    evidence
) {

    if (
        !evidence ||
        typeof evidence !== "object" ||
        Array.isArray(
            evidence
        )
    ) {

        return {
            success: false,

            text:
                "В плане отсутствует evidence"
        };

    }


    if (
        !ALLOWED_EVIDENCE_MODES.has(
            evidence.mode
        )
    ) {

        return {
            success: false,

            text:
                `Некорректный evidence.mode: ${evidence.mode}`
        };

    }


    if (
        evidence.reason !== undefined &&
        typeof evidence.reason !== "string"
    ) {

        return {
            success: false,

            text:
                "Некорректный evidence.reason"
        };

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * VALIDATE EVIDENCE AGAINST PLAN
 * =========================================================
 */


export function validateEvidencePlan(
    plan
) {

    const structureValidation =
        validateEvidenceStructure(
            plan?.evidence
        );


    if (
        !structureValidation.success
    ) {

        return structureValidation;

    }


    const mode =
        plan.evidence.mode;


    const requiresTools =
        plan.requiresTools === true;


    const steps =
        Array.isArray(
            plan.steps
        )
            ? plan.steps
            : [];


    const usedTools =
        new Set(
            steps
                .map(
                    step =>
                        step?.tool
                )
                .filter(
                    value =>
                        typeof value === "string" &&
                        value.trim()
                )
        );


    /*
     * =====================================================
     * NONE
     * =====================================================
     */


    if (
        mode === "none"
    ) {

        /*
         * evidence=none допустим и для некоторых
         * инструментальных задач.
         *
         * Например current_time:
         * инструмент нужен для действия,
         * но web-доказательства не нужны.
         */

        return {
            success: true
        };

    }


    /*
     * =====================================================
     * EXTERNAL EVIDENCE REQUIRES TOOLS
     * =====================================================
     */


    if (
        !requiresTools
    ) {

        return {
            success: false,

            text:
                (
                    `evidence.mode=${mode}, ` +
                    "но requiresTools=false"
                )
        };

    }


    /*
     * =====================================================
     * SEARCH RESULTS
     * =====================================================
     */


    if (
        mode === "search_results"
    ) {

        if (
            !usedTools.has(
                "web_search"
            )
        ) {

            return {
                success: false,

                text:
                    (
                        "План требует evidence=search_results, " +
                        "но не содержит web_search"
                    )
            };

        }


        return {
            success: true
        };

    }


    /*
     * =====================================================
     * SOURCE CONTENT
     * =====================================================
     */


    if (
        mode === "source_content"
    ) {

        if (
            !usedTools.has(
                "web_fetch"
            )
        ) {

            return {
                success: false,

                text:
                    (
                        "План требует evidence=source_content, " +
                        "но не содержит web_fetch"
                    )
            };

        }


        return {
            success: true
        };

    }


    return {
        success: false,

        text:
            `Неизвестный evidence.mode: ${mode}`
    };

}


/*
 * =========================================================
 * EXPORT MODES
 * =========================================================
 */


export function getAllowedEvidenceModes() {

    return Array.from(
        ALLOWED_EVIDENCE_MODES
    );

}
