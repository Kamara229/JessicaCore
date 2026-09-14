import {
    analyzeUserCorrection
} from "./learningAnalyzer.js";

import {
    createLearningProposal
} from "./learningProposal.js";


/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL BUILDER
 * =========================================================
 *
 * Превращает исправление пользователя
 * в предложение обучения Jessica.
 *
 *
 * Рабочая цепочка:
 *
 * task
 * + previousAnswer
 * + correction
 * + correctedAnswer
 *        ↓
 * Learning Analyzer
 *        ↓
 * Valid Learning Analysis
 *        ↓
 * Learning Proposal
 *        ↓
 * PENDING_APPROVAL
 *
 *
 * ВАЖНО:
 *
 * даже после успешного анализа
 * Experience НЕ сохраняется.
 *
 * Пользователь должен сначала
 * подтвердить предложение обучения.
 *
 *
 * Этот модуль НЕ:
 *
 * - сохраняет Skill;
 * - создаёт версию Skill в Supabase;
 * - подтверждает Proposal;
 * - повторно выполняет задачу;
 * - изменяет Experience.
 *
 * =========================================================
 */


/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function createFailureResult({

    stage,

    error,

    analysisResult = null

} = {}) {


    return {

        success:
            false,

        stage:
            stage || "proposal",

        proposal:
            null,

        analysis:
            analysisResult?.analysis || null,

        readyForApproval:
            false,

        needsClarification:
            false,

        reusable:
            false,

        error:
            String(
                error ||
                "Не удалось создать Learning Proposal"
            )

    };

}


/*
 * =========================================================
 * BUILD LEARNING PROPOSAL
 * =========================================================
 */


export async function buildLearningProposal({

    task,

    previousAnswer = "",

    correction,

    correctedAnswer = ""

} = {}) {


    /*
     * =====================================================
     * 1. ANALYZE CORRECTION
     * =====================================================
     */


    const analysisResult =
        await analyzeUserCorrection({

            task,

            previousAnswer,

            correction,

            correctedAnswer

        });


    if (
        !analysisResult?.success ||
        !analysisResult?.analysis
    ) {

        return createFailureResult({

            stage:
                analysisResult?.stage ||
                "analysis",

            error:
                analysisResult?.error ||
                "Learning Analyzer не смог проанализировать исправление",

            analysisResult

        });

    }


    /*
     * =====================================================
     * 2. ANALYSIS
     * =====================================================
     */


    const analysis =
        analysisResult.analysis;


    const clarificationQuestions =
        Array.isArray(
            analysis.clarificationQuestions
        )
            ? analysis.clarificationQuestions
            : [];


    const reusable =
        analysis.reusable === true;


    const proposedExperience =
        analysis.proposedExperience &&
        typeof analysis.proposedExperience === "object"
            ? analysis.proposedExperience
            : null;


    /*
     * =====================================================
     * 3. CREATE PROPOSAL
     * =====================================================
     */


    let proposal;


    try {


        proposal =
            createLearningProposal({

                task,

                previousAnswer,

                correction,

                correctedAnswer,

                understanding:
                    analysis.understanding || "",

                clarificationQuestions,

                proposedExperience

            });


    } catch (error) {


        console.error(
            "Learning Proposal creation error:",
            error
        );


        return createFailureResult({

            stage:
                "proposal",

            error:
                error?.message ||
                "Не удалось создать Learning Proposal",

            analysisResult

        });

    }


    /*
     * =====================================================
     * 4. APPROVAL READINESS
     * =====================================================
     *
     * Proposal существует всегда после
     * успешного анализа.
     *
     * Но подтверждать обучение можно
     * только если:
     *
     * - исправление переиспользуемое;
     * - нет обязательных уточнений;
     * - существует proposedExperience.
     *
     * =====================================================
     */


    const needsClarification =
        clarificationQuestions.length > 0;


    const readyForApproval =
        (
            reusable &&
            !needsClarification &&
            Boolean(
                proposedExperience
            )
        );


    /*
     * =====================================================
     * 5. RESULT
     * =====================================================
     */


    return {

        success:
            true,

        stage:
            "proposal",

        proposal,

        analysis,

        reusable,

        needsClarification,

        readyForApproval,

        error:
            ""

    };

}
