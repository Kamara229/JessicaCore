import { tryCreateNativeSkillPlan } from "../planner/nativeSkillPlanner.js";
import { tryCreateKnownLicenseComparison } from "../planner/knownLicenseComparison.js";

/**
 * Two named, verified canonical sources are a narrower route than a
 * generic READY single-source search Skill. Never let the latter turn
 * a two-document comparison into one web search and repeated replans.
 * The canonical plan still fetches both primary texts on each request.
 */
export function chooseNativeSubtaskPlan({
    task, executionExperience, dependencyContext = "", planningContext = {}
} = {}) {
    const canonical = tryCreateKnownLicenseComparison({task, planningContext});
    if (canonical?.handled === true && canonical.plan) return canonical;

    return tryCreateNativeSkillPlan({
        task, executionExperience, dependencyContext, planningContext
    });
}
