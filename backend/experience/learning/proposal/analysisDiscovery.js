/*
 * =========================================================
 * JESSICA PROPOSAL ANALYSIS DISCOVERY
 * =========================================================
 */

import {
    isObject
} from "./proposalUtils.js";


export function buildProposalDiscovery({

    event,

    candidate

}) {

    const discovery =

        isObject(
            event
                ?.payload
                ?.discovery
        )

            ? event.payload.discovery

            : null;


    if(!discovery){

        return null;

    }


    return {

        resolved:
            discovery.resolved === true,

        traceId:

            discovery.traceId

            ||

            event?.traceId

            ||

            null,

        patternId:

            discovery
                ?.pattern
                ?.id

            ||

            candidate?.skillId

            ||

            null,

        validation:

            isObject(
                discovery.validation
            )

                ? discovery.validation

                : null

    };

}
