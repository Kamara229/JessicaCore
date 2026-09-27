/*
 * =========================================================
 * JESSICA EXECUTION TERMINAL v8
 * =========================================================
 *
 * Финальный слой завершения Execution Cycle.
 *
 *
 * Получает:
 *
 * Execution Context
 * Normalized Failure
 *
 *
 * Возвращает:
 *
 * Terminal Result
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - принимает Failure Decision;
 * - делает Retry;
 * - делает Replan;
 * - меняет Context.
 *
 * =========================================================
 */


import {

    buildFailureResult,

    buildNoVerifiedResult,

    buildClarificationResult

} from "./executionResult.js";


import {

    getExecutionId,

    getExecutionCounters

} from "./context/contextReader.js";









export const TERMINAL_TYPE = {


    CLARIFICATION:

        "CLARIFICATION",



    NO_VERIFIED_RESULT:

        "NO_VERIFIED_RESULT",



    FAILURE:

        "FAILURE"


};









function safeString(

    value

){

    return String(

        value || ""

    )
    .trim();

}









/*
 * =========================================================
 * NORMALIZE TERMINAL FAILURE
 * =========================================================
 */


function normalizeTerminalFailure(

    failure

){

    if(

        !failure ||

        typeof failure !== "object"

    ){

        return {

            stage:

                "execution",


            failureType:

                "unknown",


            category:

                "execution",


            reason:

                "Неизвестная ошибка",


            terminalType:

                TERMINAL_TYPE.FAILURE


        };

    }









    let terminalType =

        TERMINAL_TYPE.FAILURE;









    if(

        failure.category === "clarification"

        ||

        failure.needsClarification === true

    ){

        terminalType =

            TERMINAL_TYPE.CLARIFICATION;

    }









    if(

        failure.category === "no_verified"

        ||

        failure.noVerifiedResult === true

    ){

        terminalType =

            TERMINAL_TYPE.NO_VERIFIED_RESULT;

    }









    return {


        ...failure,



        stage:

            safeString(

                failure.stage

            )

            ||

            "execution",



        failureType:

            safeString(

                failure.failureType

            )

            ||

            "execution-failure",



        category:

            failure.category ||

            "execution",



        reason:

            safeString(

                failure.reason

            )

            ||

            "Не удалось выполнить задачу",



        terminalType


    };


}









/*
 * =========================================================
 * LOG
 * =========================================================
 */


function logTerminal(

    context,

    failure

){

    const counters =

        getExecutionCounters(

            context

        );



    console.log(

        "Jessica Terminal:",

        {

            executionId:

                getExecutionId(

                    context

                ),


            attempt:

                counters.attempt,


            terminalType:

                failure.terminalType,


            category:

                failure.category,


            failureType:

                failure.failureType


        }

    );


}









/*
 * =========================================================
 * BUILD RESULT
 * =========================================================
 */


export function buildTerminalResult(

    context,

    failure = {}

){

    const normalized =

        normalizeTerminalFailure(

            failure

        );









    logTerminal(

        context,

        normalized

    );









    if(

        normalized.terminalType ===

        TERMINAL_TYPE.CLARIFICATION

    ){

        return buildClarificationResult(

            context,

            normalized

        );

    }









    if(

        normalized.terminalType ===

        TERMINAL_TYPE.NO_VERIFIED_RESULT

    ){

        return buildNoVerifiedResult(

            context,

            normalized

        );

    }









    return buildFailureResult(

        context,

        normalized

    );


}









export function isTerminalFailure(

    failure

){

    return Boolean(

        failure &&

        typeof failure === "object"

    );

}
