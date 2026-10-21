/*
 * =========================================================
 * JESSICA CORE v3
 * =========================================================
 *
 * Центральный координатор Jessica.
 *
 *
 * Flow:
 *
 * User Task
 *      ↓
 * Memory
 *      ↓
 * Decomposition
 *      ↓
 * Execution Router
 *      ↓
 * Outer Execution Trace
 *      ↓
 * Response
 *      ↓
 * Learning
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Plan;
 * - валидирует Result;
 * - хранит Experience;
 * - создаёт Skills.
 *
 * =========================================================
 */


import {
    decomposeTask
} from "./taskDecomposer.js";

import { analyzeDocumentAttachments } from "./files/documentAnalysisFacade.js";
import { translatePreviousReply } from "./language/responseLanguageService.js";

import {
    contextualizeTask,
    attachConversationToDecomposition
} from "./chat/conversationContext.js";


import {
    buildExecutionMemoryContext
} from "./memory/executionMemoryFacade.js";


import {
    createJessicaExecutionTrace,
    recordExecutionResult,
    recordExecutionSummary,
    completeExecutionTrace
} from "./trace/executionTraceFacade.js";


import {
    executeByRoute
} from "./execution/executionRouter.js";


import {
    buildJessicaResponse
} from "./response/responseFacade.js";


import {
    processExecutionLearning
} from "./learning/learningFacade.js";


/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildCoreFailure(

    stage,

    text,

    trace

) {

    completeExecutionTrace(
        trace
    );


    return {

        success:
            false,

        stage,

        text,

        executionTrace:
            trace

    };

}


/*
 * =========================================================
 * EXECUTE TASK
 * =========================================================
 */


export async function executeJessicaTask(
    task,
    { conversation = [], attachments = [], outputLanguage = "ru" } = {}
) {

    const normalizedTask =

        typeof task === "string"

            ? task.trim()

            : "";


    const executionTrace =

        createJessicaExecutionTrace(
            normalizedTask
        );


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if(
        !normalizedTask
    ){

        return buildCoreFailure(

            "input",

            "Задача не указана",

            executionTrace

        );

    }

    /*
     * File-first route: the supplied document is already our source.
     * Public web_search/web_fetch would search for the document TITLE rather
     * than read the uploaded text. Never create Experience from a private file.
     */
    if (Array.isArray(attachments) && attachments.length > 0) {
        const documentResult = await analyzeDocumentAttachments({
            task: normalizedTask,
            attachments
        });
        completeExecutionTrace(executionTrace);
        return {
            ...documentResult,
            engine: "jessica-core",
            executionTrace,
            learning: {
                success: false,
                queued: false,
                skipped: true,
                reason: "user_provided_document"
            }
        };
    }



    // A reference to the previous reply is not a new standalone task.
    // Translate the real answer, never a fabricated document or JSON.
    const previousTranslation = await translatePreviousReply(
        normalizedTask, conversation, outputLanguage
    );
    if (previousTranslation) {
        completeExecutionTrace(executionTrace);
        return {...previousTranslation, engine:"jessica-core", executionTrace};
    }

    // Keep the canonical user request intact for Experience and Learning.
    // Only expand an ambiguous follow-up at execution/decomposition time.
    const executionTask = contextualizeTask(normalizedTask, conversation);

    /*
     * =====================================================
     * MEMORY
     * =====================================================
     */


    const memoryResult =

        await buildExecutionMemoryContext(
            normalizedTask
        );


    /*
     * =====================================================
     * DECOMPOSITION
     * =====================================================
     */


    let decompositionResult;


    try {


        decompositionResult =

            await decomposeTask({

                task:
                    executionTask,

                memoryContext:
                    memoryResult.memoryContext

            });


    }catch(error){


        console.error(

            "Jessica decomposer error:",

            error

        );


        return buildCoreFailure(

            "decomposer",

            "Jessica не смогла разобрать задачу",

            executionTrace

        );

    }


    if(
        !decompositionResult?.success
    ){

        return buildCoreFailure(

            "decomposer",

            decompositionResult?.text

            ||

            "Ошибка декомпозиции",

            executionTrace

        );

    }


    const decomposition = attachConversationToDecomposition(
        decompositionResult.decomposition,
        normalizedTask,
        conversation
    );


    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     */


    let executionResult;


    try {


        executionResult =

            await executeByRoute(
                decomposition
            );


    }catch(error){


        console.error(

            "Jessica execution router error:",

            error

        );


        return buildCoreFailure(

            "execution",

            "Ошибка выполнения задачи",

            executionTrace

        );

    }


    /*
     * =====================================================
     * OUTER TRACE UPDATE
     * =====================================================
     */


    if(
        Array.isArray(
            executionResult?.results
        )
    ){

        recordExecutionSummary(

            executionTrace,

            executionResult

        );

    }else{

        recordExecutionResult(

            executionTrace,

            executionResult

        );

    }


    /*
     * Trace должен быть полностью
     * завершён ДО Learning.
     */


    completeExecutionTrace(
        executionTrace
    );


    /*
     * =====================================================
     * RESPONSE
     * =====================================================
     */


    const response =

        await buildJessicaResponse({

            task:
                normalizedTask,

            decomposition,

            executionResult,

            executionTrace

        });


    /*
     * =====================================================
     * LEARNING
     * =====================================================
     *
     * Раньше Learning Promise
     * не ожидался.
     *
     * Теперь Queue persistence
     * завершается до возврата Response.
     *
     * =====================================================
     */


    // A referential follow-up is not a standalone, reusable Skill.
    // Its trace lacks independent task grounding; do not publish it as Experience.
    const learningResult = executionTask !== normalizedTask
        ? {
            success: false,
            queued: false,
            skipped: true,
            reason: "context_dependent_chat_followup"
        }
        : await processExecutionLearning(executionTrace);


    response.learning =
        learningResult;


    /*
     * Диагностический лог.
     *
     * Не содержит полного Candidate
     * или пользовательского контента.
     */


    console.log(

        "Jessica Learning result:",

        {

            success:
                learningResult?.success === true,

            queued:
                learningResult?.queued === true,

            reason:
                learningResult?.reason || null,

            traceStatus:
                executionTrace?.status || null,

            completed:

                executionTrace
                    ?.statistics
                    ?.completed

                ??

                executionTrace
                    ?.stats
                    ?.completed

                ??

                0,

            experienceUsed:

                executionTrace
                    ?.experienceUsage
                    ?.used === true,

            experienceSkills:

                Array.isArray(
                    executionTrace
                        ?.experienceUsage
                        ?.skills
                )

                    ? executionTrace
                        .experienceUsage
                        .skills
                        .length

                    : 0

        }

    );


    return response;

}
