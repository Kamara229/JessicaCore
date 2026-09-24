/*
 * =========================================================
 * JESSICA SINGLE TASK RESPONSE BUILDER
 * =========================================================
 *
 * Формирование ответа Jessica
 * для одной подзадачи.
 *
 *
 * Ответственность:
 *
 * - преобразовать результат выполнения;
 * - сформировать API response;
 * - прикрепить executionTrace.
 *
 *
 * НЕ содержит:
 *
 * - выполнение;
 * - Planner;
 * - Tools;
 * - Validator;
 * - Experience;
 * - Learning.
 *
 * =========================================================
 */



/*
 * =========================================================
 * BUILD SINGLE TASK RESPONSE
 * =========================================================
 */


export function buildSingleTaskResponse(
    result,
    decomposition,
    executionTrace = null
) {



    /*
     * =====================================================
     * INVALID RESULT
     * =====================================================
     */


    if (
        !result ||
        typeof result !== "object"
    ) {


        return {


            success:false,


            text:
                "Jessica получила некорректный результат выполнения.",


            engine:
                "jessica-core",


            mode:
                "single",


            stage:
                "response",


            executionTrace


        };

    }





    /*
     * =====================================================
     * COMPLETED
     * =====================================================
     */


    if (
        result.status === "COMPLETED"
    ) {


        return {


            success:true,


            text:
                result.result || "",


            engine:
                "jessica-core",


            mode:
                "single",


            validated:
                result.validated === true,


            answerSource:
                result.answerSource || "unknown",



            decomposition,



            plan:
                result.plan || null,



            toolResults:
                result.toolResults || [],



            usedTools:
                result.usedTools || [],



            experience:
                result.experience || null,



            executionTrace



        };

    }






    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     */


    if (
        result.status === "NEEDS_CLARIFICATION"
    ) {


        return {


            success:false,


            needsClarification:true,


            text:
                result.result ||
                "Для выполнения задачи требуется уточнение.",



            engine:
                "jessica-core",



            mode:
                "single",



            stage:
                result.stage || "subtask",



            decomposition,



            plan:
                result.plan || null,



            toolResults:
                result.toolResults || [],



            executionTrace



        };

    }







    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return {


        success:false,


        shouldRetry:
            result.shouldRetry === true,



        text:
            result.result ||
            "Jessica не смогла выполнить задачу.",



        engine:
            "jessica-core",



        mode:
            "single",



        stage:
            result.stage || "subtask",



        decomposition,



        plan:
            result.plan || null,



        toolResults:
            result.toolResults || [],



        executionTrace



    };


}
