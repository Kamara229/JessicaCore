import {requireContiguousExperienceHistory} from "./experienceHistoryContinuity.js";

// Minimal facade: validate immutable snapshots before deriving the next version.
export function resolveExperienceVersionState(history) {
    const latestVersion = requireContiguousExperienceHistory(
        history
    );
    if (latestVersion === 0) {
        return {exists:false,latestVersion:null,nextVersion:1,previousVersion:null};
    }
    return {
        exists:true,
        latestVersion,
        nextVersion:latestVersion+1,
        previousVersion:latestVersion
    };
}
