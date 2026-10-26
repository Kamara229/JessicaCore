import express from "express";
import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";

const ordinaryJson=express.json({limit:"1mb"});
// A single verified 256-window document requires complete source pages
// plus independently materialized overflow windows (~8-10 MiB JSON).
// Keep the larger cap confined to the authenticated solve endpoint.
const documentedSolveJson=express.json({limit:"12mb"});

/** Authenticate large document requests BEFORE reading their JSON bodies.
 * All other routes retain the previous 1 MiB limit.
 */
export function registerJessicaJsonParser(app){
    app.use((req,res,next)=>{
        if(req.method==="POST" && req.path==="/api/solve"){
            return requireJessicaAuthorization(req,res,()=>{
                documentedSolveJson(req,res,next);
            });
        }
        return ordinaryJson(req,res,next);
    });
}
