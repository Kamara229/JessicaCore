/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER v7
 * =========================================================
 *
 * Центральный координатор анализа опыта
 * после выполнения пользовательской задачи.
 *
 *
 * КРИТИЧЕСКОЕ РАЗДЕЛЕНИЕ:
 *
 * Execution completed
 *
 * !=
 *
 * Positive Learning Evidence
 *
 *
 * Positive Experience reinforcement
 * разрешён только если:
 *
 * - execution завершён;
 * - semantic outcome = result;
 * - validation не degraded;
 * - validation не failed;
 * - результат не NO_VERIFIED_RESULT.
 *
 *
 * NO_VERIFIED_RESULT:
 *
 * корректный terminal outcome,
 * но НЕ положительное reinforcement.
 *
 *
 * DEGRADED_VALIDATION:
 *
 * задача может считаться выполненной,
 * но не используется как сильное
 * autonomous learning evidence.
 *
 * =========================================================
 */


import {
    matchExperiencePattern
} from "./experiencePatternMatcher.js";


import {
    getOccurrences,
    getExamples,
    calculateSuccessRate,
    calculateExperienceConfidence,
    calculateExperienceMaturity,
    resolveExperienceMaturityLevel
} from "./experienceConfidence.js";


import {
    buildNewSkillCandidate,
    buildSkillImprovementCandidate
} from "./experienceCandidateBuilder.js";


import {
    classifyLearningOutcome,
    LEARNING_OUTCOME
} from "./learningOutcomePolicy.js";


/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
) {

    return (

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


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


function normalizeUnit(
    value
) {

    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}


/*
 * =========================================================
 * IGNORE RESULT
 * =========================================================
 */


function buildIgnoreResult({

    reason,

    analysisType = "NONE",

    metrics = null,

    metadata = {}

} = {}) {

    return {

        action:
            "IGNORE",

        reusable:
            false,

        reason:

            normalizeText(
                reason
            )

            ||

            "Нет пригодного опыта для обучения",

        skillCandidate:
            null,

        analysisType,

        discoveryRequired:
            false,

        metrics:
            metrics || null,

        metadata:

            isObject(
                metadata
            )

                ? metadata

                : {}

    };

}


/*
 * =========================================================
 * LEARNING METRICS
 * =========================================================
 */


function buildLearningMetrics({

    trace,

    matchScore = 0

}) {

    const occurrences =

        getOccurrences(
            trace
        );


    const examples =

        getExamples(
            trace
        );


    const successRate =

        calculateSuccessRate(
            examples
        );


    const normalizedMatchScore =

        normalizeUnit(
            matchScore
        );


    const confidence =

        calculateExperienceConfidence({

            matchScore:
                normalizedMatchScore,

            successRate,

            occurrences

        });


    const maturity =

        calculateExperienceMaturity(
            occurrences
        );


    const maturityLevel =

        resolveExperienceMaturityLevel(
            occurrences
        );


    return {

        /*
         * examples используются локально
         * для расчёта метрик, но НЕ входят
         * в Analysis transport.
         *
         * Канонические Examples находятся
         * только в Skill Candidate.
         */

        occurrences,

        successRate,

        confidence,

        maturity,

        maturityLevel,

        matchScore:
            normalizedMatchScore

    };

}


/*
 * =========================================================
 * EXPERIENCE USAGE
 * =========================================================
 */


function getExperienceUsage(
    trace
) {

    if(
        !isObject(
            trace?.experienceUsage
        )
    ){

        return null;

    }


    return trace.experienceUsage;

}


function getUsedSkills(
    trace
) {

    const usage =

        getExperienceUsage(
            trace
        );


    if(
        !usage
        ||
        usage.used !== true
    ){

        return [];

    }


    if(
        !Array.isArray(
            usage.skills
        )
    ){

        return [];

    }


    return usage.skills.filter(

        item =>

            item
            &&
            typeof item === "object"

    );

}


/*
 * =========================================================
 * NON-POSITIVE OUTCOME
 * =========================================================
 */


function analyzeNonPositiveOutcome(
    trace,
    classification
) {

    const commonMetadata = {

        learningOutcome:

            classification.outcome,

        outcomeType:

            classification.outcomeType,

        degraded:

            classification.degraded === true,

        validationMode:

            classification.validationMode,

        completed:

            classification.completed === true,

        positiveReinforcementAllowed:

            false,

        experienceUsed:

            getExperienceUsage(trace)?.used === true

    };


    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     *
     * Это НЕ failure.
     *
     * Jessica могла правильно установить,
     * что искомого подтверждённого результата
     * в evidence нет.
     *
     * Пока сохраняем как neutral outcome
     * и не усиливаем Skill.
     *
     * В будущем может использоваться
     * отдельным Outcome Learning.
     * =====================================================
     */


    if(
        classification.outcome ===
        LEARNING_OUTCOME.NO_VERIFIED_RESULT
    ){

        return buildIgnoreResult({

            reason:

                "NO_VERIFIED_RESULT не используется как положительное Experience reinforcement",

            analysisType:

                "NO_VERIFIED_RESULT",

            metadata: {

                ...commonMetadata,

                neutralLearningEvidence:
                    true,

                failureLearningRequired:
                    false

            }

        });

    }


    /*
     * =====================================================
     * DEGRADED VALIDATION
     * =====================================================
     */


    if(
        classification.outcome ===
        LEARNING_OUTCOME.DEGRADED_VALIDATION
    ){

        return buildIgnoreResult({

            reason:

                "Execution прошёл через degraded validation и не используется для положительного reinforcement",

            analysisType:

                "DEGRADED_VALIDATION",

            metadata: {

                ...commonMetadata,

                neutralLearningEvidence:
                    true,

                failureLearningRequired:
                    false,

                revalidationUseful:
                    true

            }

        });

    }


    /*
     * =====================================================
     * FAILURE-LIKE OUTCOMES
     * =====================================================
     */


    return buildIgnoreResult({

        reason:

            classification.reason

            ||

            "Execution не является положительным Learning Evidence",

        analysisType:

            classification.outcome

            ||

            "UNSUCCESSFUL_EXECUTION",

        metadata: {

            ...commonMetadata,

            neutralLearningEvidence:
                false,

            failureLearningRequired:

                (
                    classification.outcome ===
                    LEARNING_OUTCOME.EXECUTION_FAILED

                    ||

                    classification.outcome ===
                    LEARNING_OUTCOME.VALIDATION_FAILED
                )

        }

    });

}


/*
 * =========================================================
 * EXISTING SKILL ANALYSIS
 * =========================================================
 */


function analyzeExistingSkill(
    trace
) {

    const skills =

        getUsedSkills(
            trace
        );


    if(
        skills.length === 0
    ){

        return null;

    }


    /*
     * Skill уже был найден Resolver'ом
     * и реально использовался.
     *
     * Поэтому semantic match текущему
     * Experience = 1.
     */


    const metrics =

        buildLearningMetrics({

            trace,

            matchScore:
                1

        });


    const skillCandidate =

        buildSkillImprovementCandidate({

            skills,

            trace,

            confidence:
                metrics.confidence,

            maturity:
                metrics.maturity,

            occurrences:
                metrics.occurrences

        });


    if(
        !skillCandidate
    ){

        return buildIgnoreResult({

            reason:

                "Existing Experience использовался, но Improvement Candidate не сформирован",

            analysisType:

                "EXISTING_SKILL_ERROR",

            metrics,

            metadata: {

                experienceUsed:
                    true,

                usedSkills:
                    skills.length

            }

        });

    }


    return {

        action:
            "SKILL_IMPROVEMENT",

        reusable:
            true,

        reason:

            "Существующий Experience Skill получил новый подтверждённый положительный опыт",

        skillCandidate,

        analysisType:
            "EXISTING_SKILL",

        improvementType:
            "EVIDENCE_REINFORCEMENT",

        discoveryRequired:
            false,

        metrics,

        metadata: {

            experienceUsed:
                true,

            experienceSource:

                normalizeText(
                    trace
                        ?.experienceUsage
                        ?.source
                )

                ||

                null,

            usedSkills:
                skills.length,

            targetSkillId:

                skillCandidate.targetSkillId

                ||

                skills[0]?.id

                ||

                skills[0]?.skillId

                ||

                null,

            positiveReinforcementAllowed:
                true

        }

    };

}


/*
 * =========================================================
 * KNOWN PATTERN ANALYSIS
 * =========================================================
 */


function analyzeKnownPattern(
    trace
) {

    const matched =

        matchExperiencePattern(
            trace?.task
        );


    if(
        !matched
        ||
        !matched.pattern
    ){

        return null;

    }


    const metrics =

        buildLearningMetrics({

            trace,

            matchScore:

                matched.matchScore

        });


    const skillCandidate =

        buildNewSkillCandidate({

            pattern:
                matched.pattern,

            trace,

            confidence:
                metrics.confidence,

            maturity:
                metrics.maturity,

            occurrences:
                metrics.occurrences

        });


    if(
        !skillCandidate
    ){

        return buildIgnoreResult({

            reason:

                "Known Experience Pattern найден, но NEW_SKILL Candidate не сформирован",

            analysisType:

                "KNOWN_PATTERN_ERROR",

            metrics,

            metadata: {

                patternId:

                    matched
                        ?.pattern
                        ?.id

                    ||

                    null

            }

        });

    }


    return {

        action:
            "NEW_SKILL",

        reusable:
            true,

        reason:

            "Обнаружен известный переиспользуемый сценарий с подтверждённым положительным результатом",

        skillCandidate,

        analysisType:
            "KNOWN_PATTERN",

        discoveryRequired:
            false,

        metrics,

        metadata: {

            patternId:

                matched
                    ?.pattern
                    ?.id

                ||

                null,

            patternName:

                matched
                    ?.pattern
                    ?.name

                ||

                null,

            matchedKeywords:

                Array.isArray(
                    matched.matchedKeywords
                )

                    ? matched.matchedKeywords

                    : [],

            matchScore:
                metrics.matchScore,

            positiveReinforcementAllowed:
                true

        }

    };

}


/*
 * =========================================================
 * PATTERN DISCOVERY
 * =========================================================
 */


function analyzePatternDiscovery(
    trace
) {

    const metrics =

        buildLearningMetrics({

            trace,

            matchScore:
                0

        });


    return {

        action:
            "PATTERN_DISCOVERY",

        reusable:
            false,

        reason:

            "Положительный Execution не соответствует известному Experience Pattern; требуется самостоятельное извлечение нового Pattern",

        skillCandidate:
            null,

        analysisType:
            "PATTERN_DISCOVERY",

        discoveryRequired:
            true,

        metrics,

        metadata: {

            traceId:

                trace?.id

                ||

                null,

            task:

                normalizeText(
                    trace?.task
                ),

            existingExperienceUsed:
                false,

            knownPatternMatched:
                false,

            positiveReinforcementAllowed:
                true

        }

    };

}


/*
 * =========================================================
 * MAIN ANALYSIS
 * =========================================================
 */


export function analyzeExecutionTrace(
    trace
) {

    /*
     * =====================================================
     * 1. VALIDATE TRACE
     * =====================================================
     */


    if(
        !isObject(
            trace
        )
    ){

        return buildIgnoreResult({

            reason:
                "Execution Trace отсутствует",

            analysisType:
                "INVALID_TRACE"

        });

    }


    /*
     * =====================================================
     * 2. VALIDATE TASK
     * =====================================================
     */


    if(
        !normalizeText(
            trace?.task
        )
    ){

        return buildIgnoreResult({

            reason:

                "Execution Trace не содержит Task",

            analysisType:

                "INVALID_TRACE"

        });

    }


    /*
     * =====================================================
     * 3. LEARNING OUTCOME POLICY
     * =====================================================
     */


    const classification =

        classifyLearningOutcome(
            trace
        );


    /*
     * Execution может быть корректно завершён,
     * но не являться положительным Evidence.
     *
     * Например:
     *
     * NO_VERIFIED_RESULT
     * DEGRADED_VALIDATION
     */


    if(
        classification.positive !== true
    ){

        return analyzeNonPositiveOutcome(

            trace,

            classification

        );

    }


    /*
     * =====================================================
     * 4. EXISTING EXPERIENCE
     * =====================================================
     */


    const existing =

        analyzeExistingSkill(
            trace
        );


    if(
        existing
    ){

        return existing;

    }


    /*
     * =====================================================
     * 5. KNOWN SEED PATTERN
     * =====================================================
     */


    const knownPattern =

        analyzeKnownPattern(
            trace
        );


    if(
        knownPattern
    ){

        return knownPattern;

    }


    /*
     * =====================================================
     * 6. DYNAMIC PATTERN DISCOVERY
     * =====================================================
     */


    return analyzePatternDiscovery(
        trace
    );

}
