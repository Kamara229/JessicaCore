/*
 * =========================================================
 * JESSICA LEARNING QUEUE STORAGE v4
 * =========================================================
 *
 * Persistent Storage для Learning Queue.
 *
 *
 * Flow:
 *
 * Learning Event
 *        ↓
 * Queue Item
 *        ↓
 * Supabase learning_queue
 *        ↓
 * Learning Worker
 *
 *
 * Поддерживаемые Actions:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * PATTERN_DISCOVERY
 *
 *
 * Lifecycle:
 *
 * PENDING
 *      ↓
 *
 * ┌─────────────────────────────┐
 * │ PROPOSED                    │
 * │ IGNORED                     │
 * │ FAILED                      │
 * └─────────────────────────────┘
 *
 *
 * В будущем также допускаются:
 *
 * PROCESSING
 * APPROVED
 * REJECTED
 *
 *
 * Ответственность:
 *
 * - сохранить Queue Item;
 * - получить Pending Items;
 * - нормализовать DB Row
 *   в Runtime Queue Item;
 * - изменить Queue Status;
 * - сохранить processed_at.
 *
 *
 * НЕ:
 *
 * - анализирует Experience;
 * - вызывает AI;
 * - создаёт Proposal;
 * - создаёт Skill;
 * - принимает Learning Decision;
 * - выполняет AUTO_APPROVE.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";





const TABLE_NAME =
    "learning_queue";


const MAX_EVENT_BYTES =
    512 * 1024;


const QUEUE_WRITE_ATTEMPTS =
    2;


const QUEUE_WRITE_TIMEOUT_MS =
    15000;


const QUEUE_RETRY_DELAY_MS =
    750;





/*
 * =========================================================
 * ACTIONS
 * =========================================================
 */


const ALLOWED_ACTIONS = [

    "NEW_SKILL",

    "SKILL_IMPROVEMENT",

    "PATTERN_DISCOVERY"

];





/*
 * =========================================================
 * STATUSES
 * =========================================================
 */


const ALLOWED_STATUSES = [

    "PENDING",

    "PROCESSING",

    "PROPOSED",

    "IGNORED",

    "FAILED",

    "APPROVED",

    "REJECTED"

];





/*
 * =========================================================
 * CLIENT
 * =========================================================
 */


function getClient()
{

    return getSupabaseClient();

}


/*
 * =========================================================
 * STORAGE SAFETY
 * =========================================================
 */


function getJsonBytes(
    value
) {

    try {

        return Buffer.byteLength(

            JSON.stringify(
                value ?? null
            ),

            "utf8"

        );

    }catch(error){

        return Number.POSITIVE_INFINITY;

    }

}


function sleep(
    milliseconds
) {

    return new Promise(

        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )

    );

}


function withTimeout(
    promise,
    milliseconds
) {

    let timer = null;


    const timeout =

        new Promise(

            (_, reject) => {

                timer =

                    setTimeout(

                        () => {

                            const error =
                                new Error(
                                    "Learning Queue storage timeout"
                                );


                            error.code =
                                "LEARNING_QUEUE_TIMEOUT";


                            reject(
                                error
                            );

                        },

                        milliseconds

                    );

            }

        );


    return Promise
        .race([
            promise,
            timeout
        ])
        .finally(
            () => {

                if(timer){

                    clearTimeout(
                        timer
                    );

                }

            }
        );

}


function getStorageErrorMessage(
    error
) {

    return String(

        error?.message

        ||

        error?.error

        ||

        error

        ||

        ""

    )
        .trim();

}


function isRetryableStorageError(
    error
) {

    const message =

        getStorageErrorMessage(
            error
        )
            .toLowerCase();


    const code =

        String(
            error?.code || ""
        )
            .trim()
            .toUpperCase();


    return (

        code === "57014"

        ||

        code === "LEARNING_QUEUE_TIMEOUT"

        ||

        message.includes(
            "bad gateway"
        )

        ||

        message.includes(
            "gateway timeout"
        )

        ||

        message.includes(
            "statement timeout"
        )

        ||

        message.includes(
            "service unavailable"
        )

        ||

        message.includes(
            "fetch failed"
        )

        ||

        message.includes(
            "network"
        )

        ||

        message.includes(
            "connection"
        )

    );

}


async function writeQueuePayload(
    payload
) {

    let lastError =
        null;


    for(
        let attempt = 1;

        attempt <= QUEUE_WRITE_ATTEMPTS;

        attempt++
    ){

        try {

            const response =

                await withTimeout(

                    getClient()

                        .from(
                            TABLE_NAME
                        )

                        /*
                         * Queue Item id стабилен в рамках
                         * одного Learning Event.
                         *
                         * upsert делает повтор после timeout /
                         * Bad Gateway идемпотентным.
                         */

                        .upsert(

                            payload,

                            {
                                onConflict:
                                    "id"
                            }

                        )

                        .select()

                        .single(),

                    QUEUE_WRITE_TIMEOUT_MS

                );


            if(
                !response?.error
            ){

                return {

                    success:
                        true,

                    data:
                        response?.data || null,

                    attempt

                };

            }


            lastError =
                response.error;

        }catch(error){

            lastError =
                error;

        }


        console.error(

            "Jessica Learning Queue write attempt failed:",

            {

                attempt,

                maxAttempts:
                    QUEUE_WRITE_ATTEMPTS,

                retryable:

                    isRetryableStorageError(
                        lastError
                    ),

                error:

                    getStorageErrorMessage(
                        lastError
                    )

            }

        );


        if(
            attempt >= QUEUE_WRITE_ATTEMPTS

            ||

            !isRetryableStorageError(
                lastError
            )
        ){

            break;

        }


        await sleep(
            QUEUE_RETRY_DELAY_MS
        );

    }


    return {

        success:
            false,

        error:
            lastError

    };

}





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


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





function normalizeObject(
    value
) {

    if(
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ){

        return {};

    }


    return value;

}





function normalizeAction(
    value
) {

    const action =

        normalizeText(
            value
        )
        .toUpperCase();



    return ALLOWED_ACTIONS.includes(
        action
    )

        ? action

        : null;

}





function normalizeStatus(
    value
) {

    const status =

        normalizeText(
            value
        )
        .toUpperCase();



    return ALLOWED_STATUSES.includes(
        status
    )

        ? status

        : null;

}





/*
 * =========================================================
 * VALIDATE QUEUE ITEM
 * =========================================================
 */


function isValidQueueItem(
    item
) {

    if(
        !item ||
        typeof item !== "object"
    ){

        return false;

    }



    return Boolean(

        normalizeAction(
            item.action
        )

    );

}





/*
 * =========================================================
 * NORMALIZE DATABASE ROW
 * =========================================================
 *
 * Supabase:
 *
 * skill_id
 * event_json
 * created_at
 * processed_at
 *
 *
 * Runtime:
 *
 * skillId
 * event
 * createdAt
 * processedAt
 *
 *
 * Для обратной совместимости
 * оставляем также event_json.
 *
 * =========================================================
 */


function normalizeDatabaseItem(
    row
) {

    if(
        !row ||
        typeof row !== "object"
    ){

        return null;

    }



    const event =

        normalizeObject(

            row.event_json

            ??

            row.event

        );



    return {


        /*
         * Runtime contract
         */


        id:

            row.id ||

            null,



        skillId:

            normalizeText(

                row.skill_id

                ??

                row.skillId

            )

            ||

            null,



        action:

            normalizeAction(
                row.action
            )

            ||

            normalizeText(
                row.action
            )

            ||

            null,



        confidence:

            normalizeNumber(
                row.confidence
            ),



        status:

            normalizeStatus(
                row.status
            )

            ||

            normalizeText(
                row.status
            )

            ||

            null,



        event,



        createdAt:

            row.created_at

            ??

            row.createdAt

            ??

            null,



        processedAt:

            row.processed_at

            ??

            row.processedAt

            ??

            null,



        /*
         * DB compatibility
         */


        skill_id:

            row.skill_id

            ??

            null,



        event_json:

            event,



        created_at:

            row.created_at

            ??

            null,



        processed_at:

            row.processed_at

            ??

            null

    };

}





/*
 * =========================================================
 * BUILD DATABASE PAYLOAD
 * =========================================================
 */


function buildInsertPayload(
    item
) {

    const action =

        normalizeAction(
            item.action
        );



    if(
        !action
    ){

        return null;

    }



    const status =

        normalizeStatus(
            item.status
        )

        ||

        "PENDING";



    return {


        id:

            item.id ||

            null,



        skill_id:

            normalizeText(
                item.skillId
            )

            ||

            null,



        action,



        confidence:

            normalizeNumber(
                item.confidence
            ),



        status,



        event_json:

            normalizeObject(
                item.event
            ),



        created_at:

            item.createdAt

            ||

            new Date()
                .toISOString()

    };

}





/*
 * =========================================================
 * SAVE QUEUE ITEM
 * =========================================================
 */


export async function saveLearningQueueItem(
    item
) {


    if(
        !isValidQueueItem(
            item
        )
    ){

        return {


            success:
                false,


            error:
                "Invalid learning queue item"

        };

    }



    const payload =

        buildInsertPayload(
            item
        );



    if(
        !payload
    ){

        return {


            success:
                false,


            error:
                "Learning Queue payload не сформирован"

        };

    }



    const eventBytes =

        getJsonBytes(
            payload.event_json
        );


    if(
        !Number.isFinite(
            eventBytes
        )

        ||

        eventBytes > MAX_EVENT_BYTES
    ){

        console.error(

            "Jessica Learning Queue payload rejected:",

            {

                id:
                    payload.id || null,

                action:
                    payload.action || null,

                eventBytes,

                maxEventBytes:
                    MAX_EVENT_BYTES

            }

        );


        return {

            success:
                false,

            error:

                "Learning Queue event payload слишком большой"

        };

    }


    try {


        const writeResult =

            await writeQueuePayload(
                payload
            );


        if(
            writeResult?.success !== true
        ){

            const error =

                writeResult?.error

                ||

                new Error(
                    "Learning Queue write failed"
                );


            console.error(

                "Jessica Learning Queue insert error:",

                error

            );


            return {

                success:
                    false,

                error:

                    getStorageErrorMessage(
                        error
                    )

                    ||

                    "Learning Queue write failed"

            };

        }


        const data =

            writeResult.data;


        const normalized =

            normalizeDatabaseItem(
                data
            );


        console.log(

            "Jessica Learning Queue stored:",

            {

                id:

                    normalized?.id

                    ||

                    data?.id

                    ||

                    null,

                action:
                    payload.action,

                eventBytes,

                attempt:
                    writeResult.attempt

            }

        );


        return {


            success:
                true,


            id:

                normalized?.id

                ||

                data?.id

                ||

                null,


            item:

                normalized

                ||

                data,

            eventBytes,

            attempt:
                writeResult.attempt

        };



    }catch(error){


        console.error(

            "Jessica Learning Queue storage error:",

            error

        );



        return {


            success:
                false,


            error:

                error?.message

                ||

                "Storage error"

        };

    }

}





/*
 * =========================================================
 * GET PENDING ITEMS
 * =========================================================
 */


export async function getPendingLearningItems()
{

    try {


        const {
            data,
            error
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .select("*")

                .eq(
                    "status",
                    "PENDING"
                )

                .order(

                    "created_at",

                    {
                        ascending:
                            true
                    }

                );



        if(
            error
        ){

            return {


                success:
                    false,


                items:
                    [],


                error:
                    error.message

            };

        }



        const rows =

            Array.isArray(
                data
            )

                ? data

                : [];



        const items =

            rows

                .map(
                    normalizeDatabaseItem
                )

                .filter(
                    Boolean
                );



        return {


            success:
                true,


            items,


            count:

                items.length

        };



    }catch(error){


        return {


            success:
                false,


            items:
                [],


            error:

                error?.message

                ||

                "Storage error"

        };

    }

}





/*
 * =========================================================
 * UPDATE STATUS
 * =========================================================
 */


export async function updateLearningQueueItemStatus(

    id,

    status

) {


    const normalizedId =

        normalizeText(
            id
        );



    if(
        !normalizedId
    ){

        return {


            success:
                false,


            error:
                "Queue item id required"

        };

    }



    const normalizedStatus =

        normalizeStatus(
            status
        );



    if(
        !normalizedStatus
    ){

        return {


            success:
                false,


            error:

                `Unsupported Queue status: ${normalizeText(status)}`

        };

    }



    const payload = {


        status:

            normalizedStatus

    };



    /*
     * PENDING означает ещё не обработан.
     *
     * Для остальных состояний
     * фиксируем время обработки.
     */


    if(
        normalizedStatus !==
        "PENDING"
    ){

        payload.processed_at =

            new Date()
                .toISOString();

    }



    try {


        const {
            data,
            error
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .update(
                    payload
                )

                .eq(
                    "id",
                    normalizedId
                )

                .select()
                .single();



        if(
            error
        ){

            console.error(

                "Jessica Learning Queue status update error:",

                error

            );



            return {


                success:
                    false,


                error:
                    error.message

            };

        }



        return {


            success:
                true,


            item:

                normalizeDatabaseItem(
                    data
                )

                ||

                data

        };



    }catch(error){


        return {


            success:
                false,


            error:

                error?.message

                ||

                "Storage error"

        };

    }

}





/*
 * =========================================================
 * ACTION SUPPORT
 * =========================================================
 */


export function isSupportedLearningQueueAction(
    action
) {

    return Boolean(

        normalizeAction(
            action
        )

    );

}





/*
 * =========================================================
 * STATUS SUPPORT
 * =========================================================
 */


export function isSupportedLearningQueueStatus(
    status
) {

    return Boolean(

        normalizeStatus(
            status
        )

    );

}
