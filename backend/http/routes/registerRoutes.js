import {
    registerRootRoute
} from "./rootRoute.js";

import {
    registerHealthRoute
} from "./healthRoute.js";

import {
    registerSearchRoute
} from "./searchRoute.js";

import {
    registerFetchRoute
} from "./fetchRoute.js";

import {
    registerSolveRoute
} from "./solveRoute.js";
import { registerDocumentExtractRoute } from "./documentExtractRoute.js";
import {registerChatTitleRoute} from "./chatTitleRoute.js";
import {registerChatDebugArchiveRoute} from "./chatDebugArchiveRoute.js";

import {
    registerLearningTestRoute
} from "./learningTestRoute.js";

import {
    registerNotFoundRoute
} from "./notFoundRoute.js";


/*
 * =========================================================
 * JESSICA ROUTES
 * =========================================================
 *
 * Центральный координатор HTTP маршрутов.
 *
 *
 * server.js
 *      ↓
 * registerRoutes.js
 *      ↓
 * отдельные route-модули
 *
 *
 * ВАЖНО:
 *
 * 404 регистрируется последним.
 *
 * =========================================================
 */


export function registerRoutes(
    app
) {


    /*
     * =====================================================
     * PUBLIC / SYSTEM
     * =====================================================
     */


    registerRootRoute(
        app
    );


    registerHealthRoute(
        app
    );


    /*
     * =====================================================
     * TOOLS
     * =====================================================
     */


    registerSearchRoute(
        app
    );


    registerFetchRoute(
        app
    );


    /*
     * =====================================================
     * JESSICA CORE
     * =====================================================
     */


    registerDocumentExtractRoute(app);
    registerChatTitleRoute(app);
    registerChatDebugArchiveRoute(app);
    registerSolveRoute(
        app
    );


    /*
     * =====================================================
     * LEARNING TEST
     * =====================================================
     *
     * Временный защищённый маршрут:
     *
     * POST /api/learning/test
     *
     * Пока умеет только создавать
     * Learning Proposal.
     *
     * Experience не сохраняет.
     *
     * =====================================================
     */


    registerLearningTestRoute(
        app
    );


    /*
     * =====================================================
     * 404
     * =====================================================
     *
     * Всегда последний.
     *
     * =====================================================
     */


    registerNotFoundRoute(
        app
    );


}
