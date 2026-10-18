/*
 * =========================================================
 * JESSICA CANDIDATE EXAMPLE MERGE
 * =========================================================
 */


import {
    normalizeText
} from "../candidateUtils.js";


import {
    compactCandidateExample
} from "./candidateExampleCompact.js";


function buildExampleKey(
    example
) {

    if(
        !example
        ||
        typeof example !== "object"
        ||
        Array.isArray(example)
    ){

        return "";

    }


    if(
        example.traceId
    ){

        return (
            "trace:"
            +
            String(
                example.traceId
            )
        );

    }


    const task =

        normalizeText(
            example.task
        );


    let result = "";


    try {

        result =

            JSON.stringify(
                example.result ?? null
            );

    }catch(error){

        result =

            String(
                example.result || ""
            );

    }


    return (
        task
        +
        "::"
        +
        result
    );

}


export function mergeCandidateExamples(
    ...exampleSets
) {

    const result = [];

    const known =
        new Set();


    for(
        const set
        of exampleSets
    ){

        if(
            !Array.isArray(set)
        ){

            continue;

        }


        for(
            const rawExample
            of set
        ){

            const example =

                compactCandidateExample(
                    rawExample
                );


            if(!example){

                continue;

            }


            const key =

                buildExampleKey(
                    example
                );


            if(
                key
                &&
                known.has(
                    key
                )
            ){

                continue;

            }


            if(key){

                known.add(
                    key
                );

            }


            result.push(
                example
            );

        }

    }


    return result;

}
