/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER v5
 * =========================================================
 *
 * Центральный координатор анализа опыта
 * после выполнения пользовательской задачи.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Validate Execution
 *        ↓
 *
 * Existing Experience used?
 *
 * YES
 *  ↓
 * SKILL_IMPROVEMENT
 *
 *
 * NO
 *  ↓
 * Known Experience Pattern?
 *
 * YES
 *  ↓
 * NEW_SKILL
 *
 *
 * NO
 *  ↓
 * PATTERN_DISCOVERY
 *  ↓
 * Learning Queue
 *  ↓
 * Background AI Pattern Extraction
 *
 *
 * Возможные Actions:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * PATTERN_DISCOVERY
 * IGNORE
 *
 *
 * Ответственность:
 *
 * - принять Execution Trace;
 * - определить успешность Execution;
 * - определить использование Existing Experience;
 * - найти известный Experience Pattern;
 * - рассчитать Learning Metrics;
 * - сформировать Learning Candidate;
 * - определить необходимость Pattern Discovery.
 *
 *
 * НЕ:
 *
 * - вызывает AI;
 * - сохраняет Experience;
 * - работает с Supabase;
 * - принимает AUTO_APPROVE;
 * - создаёт версии Skill;
 * - накапливает KEEP_CANDIDATE;
 * - выполняет Pattern Extraction.
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





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

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
 * TRACE SUCCESS
 * =========================================================
 *
 * Execution Trace может приходить
 * из разных уровней Runtime.
 *
 * Поэтому не привязываемся только
 * к одному полю stats.completed.
 *
 * =========================================================
 */


function isSuccessfulTrace(
    trace
) {


    /*
     * Terminal Execution Result
     */


    if(
        typeof trace?.result?.success ===
        "boolean"
    ){

        return trace.result.success;

    }



    /*
     * Direct success
     */


    if(
        typeof trace?.success ===
        "boolean"
    ){

        return trace.success;

    }



    /*
     * Legacy / outer statistics
     */


    const completed =

        normalizeNumber(

            trace?.stats?.completed

            ??

            trace?.statistics?.completed

        );



    if(
        completed > 0
    ){

        return true;

    }



    /*
     * Terminal / context status
     */


    const status =

        normalizeText(

            trace?.result?.status

            ||

            trace?.status

        )
        .toUpperCase();



    if(
        status === "COMPLETED"
    ){

        return true;

    }



    /*
     * NO_VERIFIED_RESULT в новой модели
     * может быть корректным успешным
     * semantic outcome.
     *
     * Если Result уже содержит success=true,
     * мы попадём в первый branch.
     *
     * Этот fallback оставляем для
     * совместимости переходного периода.
     */


    if(
        status === "NO_VERIFIED_RESULT"
        &&
        trace?.result?.validation?.valid === true
    ){

        return true;

    }



    return false;

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


        occurrences,


        examples,


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
        !usage ||
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

            item &&

            typeof item === "object"

    );

}





/*
 * =========================================================
 * EXISTING SKILL ANALYSIS
 * =========================================================
 *
 * Если Experience уже участвовал
 * в успешном Execution,
 * новый Execution становится
 * дополнительным evidence.
 *
 *
 * На текущем этапе:
 *
 * SKILL_IMPROVEMENT
 *
 * означает прежде всего
 * EVIDENCE_REINFORCEMENT.
 *
 *
 * Позже отдельный Semantic Improvement
 * Analyzer сможет изменять:
 *
 * - workflow;
 * - constraints;
 * - validationRules;
 * - failurePatterns;
 * - strategy.
 *
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
     * Skill уже был найден
     * Experience Resolver'ом
     * и реально использовался.
     *
     * Поэтому semantic match
     * текущему Experience = 1.
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

            "Существующий Experience Skill получил новый подтверждённый опыт",



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

                null

        }

    };

}





/*
 * =========================================================
 * KNOWN PATTERN ANALYSIS
 * =========================================================
 *
 * EXPERIENCE_PATTERNS являются
 * seed knowledge.
 *
 * Это быстрый deterministic path,
 * но НЕ граница способности Jessica
 * к обучению.
 *
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
        !matched ||
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

            "Обнаружен известный переиспользуемый сценарий, пригодный для Experience Skill",



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

                    ?

                    matched.matchedKeywords

                    :

                    [],



            matchScore:

                metrics.matchScore

        }

    };

}





/*
 * =========================================================
 * PATTERN DISCOVERY
 * =========================================================
 *
 * Успешный Execution:
 *
 * - Existing Experience не использовался;
 * - Known Pattern не найден.
 *
 *
 * Раньше такой опыт:
 *
 * IGNORE
 *
 *
 * Теперь:
 *
 * PATTERN_DISCOVERY
 *
 *
 * Дальше:
 *
 * Learning Router
 *      ↓
 * Queue
 *      ↓
 * Background Pattern Discovery Worker
 *      ↓
 * AI Experience Pattern Extractor
 *
 *
 * reusable=false здесь намеренно.
 *
 * Мы ещё НЕ доказали,
 * что Execution действительно содержит
 * reusable Skill.
 *
 * Это должен решить Pattern Extractor.
 *
 * =========================================================
 */


function analyzePatternDiscovery(
    trace
) {


    /*
     * Pattern пока неизвестен,
     * поэтому matchScore = 0.
     *
     * После успешного AI extraction
     * Pattern Discovery Worker
     * пересчитает Confidence
     * уже с matchScore = 1.
     */


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

            "Успешный Execution не соответствует известному Experience Pattern; требуется самостоятельное извлечение нового Pattern",



        skillCandidate:

            null,



        analysisType:

            "PATTERN_DISCOVERY",



        discoveryRequired:

            true,



        metrics,



        metadata: {


            traceId:

                trace?.id ||

                null,



            task:

                normalizeText(
                    trace?.task
                ),



            existingExperienceUsed:

                false,



            knownPatternMatched:

                false

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
     * 3. SUCCESS CHECK
     * =====================================================
     *
     * Пока активный Learning path
     * создаёт Experience только
     * из успешных executions.
     *
     *
     * Неудачные executions НЕ считаются
     * бесполезными.
     *
     * В будущем они должны идти
     * в отдельный Failure Learning:
     *
     * - failurePatterns;
     * - avoidPatterns;
     * - constraints;
     * - strategy corrections.
     *
     * =====================================================
     */


    if(
        !isSuccessfulTrace(
            trace
        )
    ){

        return buildIgnoreResult({

            reason:

                "Execution не подтверждён как успешный",


            analysisType:

                "UNSUCCESSFUL_EXECUTION",


            metadata: {

                failureLearningRequired:

                    true

            }

        });

    }



    /*
     * =====================================================
     * 4. EXISTING EXPERIENCE
     * =====================================================
     *
     * Existing Skill имеет максимальный
     * приоритет.
     *
     * Если Jessica уже использовала
     * Experience для решения задачи,
     * новый результат должен развивать
     * именно эту Experience lineage,
     * а не создавать дублирующий Skill.
     *
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
     *
     * Новый тип успешной задачи
     * больше НЕ выбрасывается.
     *
     * Он отправляется в фоновый
     * AI Pattern Extraction.
     *
     * =====================================================
     */


    return analyzePatternDiscovery(
        trace
    );

        }
