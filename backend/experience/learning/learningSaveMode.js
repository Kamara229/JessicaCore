// The persistence mode reflects an immutable version chain, not the proposal action.
export function resolveLearningSaveMode(state) {
    if (state?.exists === false &&
        state.nextVersion === 1 && state.previousVersion === null) {
        return "create";
    }
    if (state?.exists === true &&
        Number.isInteger(state.previousVersion) &&
        state.previousVersion > 0 &&
        state.nextVersion === state.previousVersion + 1) {
        return "update";
    }
    throw new Error("Invalid Experience version state");
}
