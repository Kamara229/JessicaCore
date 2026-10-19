import { extractNoticeDutyExcerpts } from "./noticeDutyExcerpts.js";
import { extractPatentClauseExcerpts } from "./patentClauseExcerpts.js";
import { selectOperativeSourceText } from "./operativeSourceScope.js";

/*
 * =========================================================
 * JESSICA CLAIM EVIDENCE CONTEXT v1
 * =========================================================
 *
 * Строит компактный Source Context
 * для Claim Evidence Validator.
 *
 *
 * Проблема:
 *
 * web_fetch может вернуть десятки тысяч
 * символов текста.
 *
 * Передавать весь документ Validator AI:
 *
 * - дорого;
 * - медленно;
 * - может превышать ITPM;
 * - ухудшает signal/noise ratio.
 *
 *
 * Решение:
 *
 * Task + Answer
 *      ↓
 * Query Terms
 *      ↓
 * Source Chunks
 *      ↓
 * Relevance Scoring
 *      ↓
 * Top Relevant Chunks
 *      ↓
 * Compact Source Context
 *
 *
 * Важно:
 *
 * Это deterministic preprocessing.
 *
 * AI здесь НЕ используется.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_SOURCE_COUNT =
    4;


const MAX_CONTEXT_LENGTH =
    10800;


const CHUNK_LENGTH =
    1200;


const CHUNK_OVERLAP =
    180;


const MAX_CHUNKS_PER_SOURCE =
    4;


const MAX_QUERY_TERMS =
    48;


/*
 * =========================================================
 * STOP WORDS
 * =========================================================
 */


const STOP_WORDS =
    new Set([

        /*
         * Russian
         */

        "это",
        "как",
        "для",
        "что",
        "или",
        "при",
        "его",
        "ее",
        "она",
        "они",
        "оно",
        "под",
        "над",
        "из",
        "до",
        "по",
        "на",
        "в",
        "во",
        "не",
        "да",
        "же",
        "бы",
        "ли",
        "найди",
        "найти",
        "укажи",
        "указать",
        "актуальную",
        "актуальный",
        "актуальное",
        "версию",
        "версия",
        "официальном",
        "официальный",
        "сайте",
        "сайт",
        "источник",
        "источника",

        /*
         * English
         */

        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "into",
        "your",
        "you",
        "its",
        "our",
        "their",
        "find",
        "official",
        "site",
        "source",
        "version"

    ]);


/*
 * =========================================================
 * BASIC HELPERS
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
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


function normalizeUrl(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


/*
 * =========================================================
 * FETCH RESULTS
 * =========================================================
 */


function findFetchResults(
    taskRunResult
) {

    const results =

        Array.isArray(
            taskRunResult?.results
        )

            ? taskRunResult.results

            : [];


    return results

        .filter(
            result =>

                result?.tool === "web_fetch"

                &&

                result?.success === true

                &&

                typeof result?.data?.content === "string"

                &&

                result.data.content.trim()
        )

        .slice(
            0,
            MAX_SOURCE_COUNT
        )
        .map(item=>({
            ...item,
            data:{
                ...item.data,
                content:selectOperativeSourceText(item.data.content)
            }
        }));

}


/*
 * =========================================================
 * QUERY TERMS
 * =========================================================
 */


function extractTerms(
    value
) {

    const text =

        normalizeText(
            value
        )
        .toLowerCase();


    if(
        !text
    ){

        return [];

    }


    const matches =

        text.match(
            /[\p{L}\p{N}][\p{L}\p{N}._+\-/]{1,}/gu
        )

        ||

        [];


    const result = [];


    for(
        const raw
        of matches
    ){

        const term =

            raw
                .trim()
                .toLowerCase();


        if(
            !term
        ){

            continue;

        }


        /*
         * Короткие числовые/version tokens
         * могут быть очень важны:
         *
         * 3.15.0
         * JDK25
         * 2026
         */

        const containsDigit =
            /\d/.test(term);


        if(
            term.length < 3
            &&
            !containsDigit
        ){

            continue;

        }


        if(
            STOP_WORDS.has(
                term
            )
        ){

            continue;

        }


        if(
            !result.includes(
                term
            )
        ){

            result.push(
                term
            );

        }


        if(
            result.length >=
            MAX_QUERY_TERMS
        ){

            break;

        }

    }


    return result;

}


function buildQueryTerms(
    task,
    answer
) {

    const combined = String(task || "") + " " + String(answer || "");
    const aliases = [];
    if (/патент|patent/iu.test(combined)) aliases.push("patent","litigation","institute","terminate","grant");
    if (/ограничен|товарн|знак|бренд|trademark/iu.test(combined)) aliases.push("trademark","endorse");
    if (/уведомлен|копирайт|copyright|notice/iu.test(combined)) aliases.push("copyright","notice","retain");
    return [...new Set([...aliases, ...extractTerms(task), ...extractTerms(answer)])]
        .slice(0, MAX_QUERY_TERMS);

}


/*
 * =========================================================
 * CHUNKS
 * =========================================================
 */


function splitIntoChunks(
    content
) {

    const text =

        normalizeText(
            content
        );


    if(
        !text
    ){

        return [];

    }


    if(
        text.length <=
        CHUNK_LENGTH
    ){

        return [
            text
        ];

    }


    const chunks = [];


    const step =

        Math.max(

            CHUNK_LENGTH
            -
            CHUNK_OVERLAP,

            200

        );


    for(
        let start = 0;
        start < text.length;
        start += step
    ){

        let end =

            Math.min(

                start
                +
                CHUNK_LENGTH,

                text.length

            );


        /*
         * Не режем последнее слово,
         * если рядом есть пробел.
         */

        if(
            end < text.length
        ){

            const lastSpace =

                text.lastIndexOf(
                    " ",
                    end
                );


            if(
                lastSpace > start + 400
            ){

                end =
                    lastSpace;

            }

        }


        const chunk =

            text
                .slice(
                    start,
                    end
                )
                .trim();


        if(
            chunk
        ){

            chunks.push({

                text:
                    chunk,

                start

            });

        }


        if(
            end >= text.length
        ){

            break;

        }

    }


    return chunks;

}


/*
 * =========================================================
 * RELEVANCE
 * =========================================================
 */


function countOccurrences(
    text,
    term
) {

    if(
        !text
        ||
        !term
    ){

        return 0;

    }


    let count = 0;

    let offset = 0;


    while(
        offset < text.length
    ){

        const index =

            text.indexOf(
                term,
                offset
            );


        if(
            index === -1
        ){

            break;

        }


        count += 1;


        offset =

            index
            +
            term.length;

    }


    return count;

}


function scoreChunk(
    chunk,
    terms,
    index
) {

    const text =

        normalizeText(
            chunk?.text
        )
        .toLowerCase();


    let score = 0;


    for(
        const term
        of terms
    ){

        const hits =

            countOccurrences(
                text,
                term
            );


        if(
            hits === 0
        ){

            continue;

        }


        const containsDigit =
            /\d/.test(term);


        score +=

            hits
            *
            (
                containsDigit

                    ? 6

                    : 2
            );

    }


    // Claims about patents need both the grant and its litigation
    // termination conditions, not just the heading "Patent License".
    if (terms.includes("patent") && /patent litigation|institute patent|terminate/.test(text)) {
        score += 28;
    }
    if (terms.includes("trademark") && /trademark|trade name|product name/.test(text)) {
        score += 12;
    }

    /*
     * Первый fragment получает небольшой bonus:
     *
     * title / release header / summary
     * часто находятся в начале страницы.
     */

    if(
        index === 0
    ){

        score +=
            1;

    }


    return score;

}


/*
 * =========================================================
 * SOURCE SELECTION
 * =========================================================
 */


function selectSourceChunks(
    result,
    terms
) {

    const chunks =

        splitIntoChunks(
            result?.data?.content
        );


    if(
        chunks.length === 0
    ){

        return [];

    }


    const scored =

        chunks.map(

            (
                chunk,
                index
            ) => ({

                ...chunk,

                index,

                score:

                    scoreChunk(

                        chunk,

                        terms,

                        index

                    )

            })

        );


    scored.sort(
        (
            left,
            right
        ) => {

            if(
                right.score !==
                left.score
            ){

                return (
                    right.score
                    -
                    left.score
                );

            }


            return (
                left.index
                -
                right.index
            );

        }
    );


    const selected =

        scored.slice(
            0,
            MAX_CHUNKS_PER_SOURCE
        );


    /*
     * В итоговом prompt удобнее видеть
     * snippets в исходном порядке страницы.
     */

    selected.sort(
        (
            left,
            right
        ) =>
            left.index
            -
            right.index
    );


    return selected;

}


/*
 * =========================================================
 * FORMAT SOURCE
 * =========================================================
 */


function formatSource(
    result,
    sourceIndex,
    chunks,
    task
) {

    const url =

        normalizeUrl(
            result?.data?.url
        );


    const title =

        normalizeText(
            result?.data?.title
        );


    const parts = [

        `SOURCE ${sourceIndex + 1}`,

        `URL: ${url}`,

        `TITLE: ${title}`

    ];


    // Prioritize actual obligations from the fetched text. These are
    // quotations, not independently verified interpretations.
    const dutyExcerpts=extractNoticeDutyExcerpts(result?.data?.content,task);
    if(dutyExcerpts.length) {
        parts.push("", "PRIMARY SOURCE DUTY EXCERPTS — verify exact obligations and conditions:");
        dutyExcerpts.forEach((quote,index)=>{
            parts.push("DUTY EXCERPT "+(index+1)+": "+quote);
        });
    }

    // Supply separate source-backed grant/termination provisions;
    // they are not interchangeable legal claims.
    extractPatentClauseExcerpts(result?.data?.content,task).forEach((quote,index)=>{
        parts.push("", "PATENT EXCERPT "+(index+1)+": "+quote);
    });

    chunks.forEach(
        (
            chunk,
            index
        ) => {

            parts.push(
                ""
            );


            parts.push(
                `SNIPPET ${index + 1}:`
            );


            parts.push(
                chunk.text
            );

        }
    );


    return parts.join(
        "\n"
    );

}


/*
 * =========================================================
 * PUBLIC CONTEXT BUILDER
 * =========================================================
 */


export function buildClaimEvidenceContext({

    task,

    answer,

    taskRunResult

} = {}) {

    const fetchResults =

        findFetchResults(
            taskRunResult
        );


    if(
        fetchResults.length === 0
    ){

        return {

            sourceText:
                "",

            sourceCount:
                0,

            chunkCount:
                0,

            sourceChars:
                0,

            queryTerms:
                []

        };

    }


    const queryTerms =

        buildQueryTerms(

            task,

            answer

        );


    const sourceParts = [];

    let chunkCount =
        0;


    for(
        let index = 0;
        index < fetchResults.length;
        index++
    ){

        const result =

            fetchResults[index];


        const chunks =

            selectSourceChunks(

                result,

                queryTerms

            );


        if(
            chunks.length === 0
        ){

            continue;

        }


        chunkCount +=
            chunks.length;


        sourceParts.push(

            formatSource(

                result,

                index,

                chunks,

                task

            )

        );

    }


    const sourceText =

        sourceParts
            .join(
                "\n\n====================\n\n"
            )
            .slice(
                0,
                MAX_CONTEXT_LENGTH
            );


    return {

        sourceText,

        sourceCount:
            fetchResults.length,

        chunkCount,

        sourceChars:
            sourceText.length,

        queryTerms

    };

  }
