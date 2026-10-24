import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

const MOZILLA="https://www.mozilla.org/en-US/MPL/2.0/";
const APACHE="https://www.apache.org/licenses/LICENSE-2.0.txt";
const flat=s=>String(s||"").replace(/\s+/g," ").toLowerCase();

function evidenceFor(url,run) {
    const results=(run?.results||[]).filter(x=>x?.tool==="web_fetch"&&x.success===true);
    if(results.length!==2)return null;
    return results.find(x=>x.data?.url===url&&!x.data.truncated)?.data?.content||null;
}

export function groundedMplPatentComparison(task,plan,run) {
    const question=String(task||"");
    if(!/(сравн|сопостав|compare)/iu.test(question) ||
       !/(патент|patent)/iu.test(question) ||
       !/(?:mpl\s*2(?:\.0)?|mozilla public licen[sc]e\s*2(?:\.0)?)/iu.test(question) ||
       !/apache(?:\s+license)?\s*2(?:\.0)?/iu.test(question) ||
       !/native_known_license_comparison/.test(plan?.intent||"") ||
       verifyPairedSourceContent(plan,run)?.valid!==true) return null;
    const mozillaSource=evidenceFor(MOZILLA,run), apacheSource=evidenceFor(APACHE,run);
    if(!mozillaSource || !apacheSource)return null;
    const m=flat(selectOperativeSourceText(mozillaSource));
    const a=flat(selectOperativeSourceText(apacheSource));
    const mplClauses=[
       "mozilla public license",
       "2.1. grants",
       "under patent claims of such contributor to make, use, sell",
       "2.3. limitations on grant scope",
       "notwithstanding section 2.1(b) above, no patent license is granted by a contributor",
       "5.1. the rights granted under this license will terminate automatically if you fail to comply",
       "5.2. if you initiate litigation against any entity by asserting a patent infringement claim",
       "excluding declaratory judgment actions, counter-claims, and cross-claims",
       "alleging that a contributor version directly or indirectly infringes any patent",
       "under section 2.1 of this license shall terminate"
    ];
    const apacheClauses=[
       "3. grant of patent license",
       "each contributor hereby grants to you a perpetual",
       "patent license to make, have made, use",
       "where such license applies only to those patent claims licensable",
       "if you institute patent litigation against any entity",
       "including a cross-claim or counterclaim in a lawsuit",
       "alleging that the work or a contribution incorporated within the work constitutes direct or contributory patent infringement",
       "any patent licenses granted to you under this license for that work shall terminate",
       "as of the date such litigation is filed"
    ];
    if(!mplClauses.every(clause=>m.includes(clause)) ||
       !apacheClauses.every(clause=>a.includes(clause)))return null;
    const text=[
       "MPL 2.0: пункт 2.1(b) прямо предоставляет неисключительную безвозмездную лицензию на патентные притязания участника в отношении его вкладов или Contributor Version; пункт 2.3 ограничивает её объём (в том числе исключает удалённый код и некоторые изменения или сочетания с другим ПО). Пункт 5.1 предусматривает автоматическое прекращение прав при нарушении условий с механизмом восстановления при устранении нарушения. По пункту 5.2 при инициировании иска о нарушении патента со стороны Contributor Version прекращаются права по пункту 2.1 от всех участников на Covered Software. Иск о признании права, встречные и перекрёстные иски прямо исключены. Официальный текст: "+MOZILLA,
       "Apache 2.0: раздел 3 прямо предоставляет патентную лицензию в пределах патентных притязаний, неизбежно затрагиваемых вкладом участника. Предоставленные по Apache патентные права на соответствующее произведение прекращаются с даты подачи иска, утверждающего, что произведение или включённый вклад нарушает патент; раздел 3 прямо включает встречные и перекрёстные иски. Это не условие о любом патентном споре. Официальный текст: "+APACHE,
       "Ключевое различие: MPL 2.0, п. 5.2, исключает встречные и перекрёстные патентные иски из указанного основания прекращения и говорит о правах по п. 2.1; Apache 2.0, раздел 3, включает такие иски и прекращает патентные права на соответствующее произведение."
    ].join("\n\n");
    return {success:true,source:"bounded-patent-comparison",text};
}
