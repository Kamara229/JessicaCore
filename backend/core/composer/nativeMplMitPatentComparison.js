import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

const MPL="https://www.mozilla.org/media/MPL/2.0/index.txt";
const MIT="https://opensource.org/license/mit";
const flat=s=>String(s||"").replace(/\s+/g," ").toLowerCase();

export function groundedMplMitPatentComparison(task,plan,run) {
    const question=String(task||"");
    if (!/(?:сравн|сопостав|compare)/iu.test(question) ||
        !/(?:патент|patent)/iu.test(question) ||
        !/(?:mpl\s*2(?:\.0)?|mozilla public licen[sc]e\s*2(?:\.0)?)/iu.test(question) ||
        !/\bmit(?:\s+license)?\b/iu.test(question) ||
        plan?.intent!=="native_known_license_comparison" ||
        verifyPairedSourceContent(plan,run)?.valid!==true)return null;

    const results=(run?.results||[]).filter(x=>x?.tool==="web_fetch"&&x.success===true);
    if(results.length!==2)return null;
    const sourceFor=url=>results.find(x=>x.data?.url===url&&x.data?.truncated!==true)?.data?.content;
    const mRaw=sourceFor(MPL),mitRaw=sourceFor(MIT);
    if(!mRaw||!mitRaw)return null;
    const m=flat(selectOperativeSourceText(mRaw)),mit=flat(mitRaw);
    const requiredMpl=[
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
    if(!requiredMpl.every(part=>m.includes(part)))return null;
    const start=mit.indexOf("permission is hereby granted, free of charge");
    const last=mit.indexOf("other dealings in the software",start);
    if(start<0||last<start+400)return null;
    const mitGrant=mit.slice(start,last+"other dealings in the software".length);
    const requiredMit=[
        "including without limitation the rights to use, copy, modify, merge, publish, distribute",
        "the above copyright notice and this permission notice shall be included",
        "the software is provided",
        "in no event shall the authors or copyright holders be liable"
    ];
    if(!requiredMit.every(part=>mitGrant.includes(part))||
        /\bpatent(?:s|able)?\b/iu.test(mitGrant))return null;

    const text=[
        "MPL 2.0: пункт 2.1(b) прямо предоставляет патентную лицензию в отношении вкладов участника или Contributor Version; пункт 2.3 ограничивает её объём. Пункт 5.1 предусматривает прекращение прав при несоблюдении лицензии и условия восстановления. Отдельно пункт 5.2 прекращает права, предоставленные всеми участниками по пункту 2.1 на Covered Software, когда получатель инициирует патентный иск о нарушении со стороны Contributor Version. Из этого основания исключены иски о признании права, встречные и перекрёстные иски. Официальный текст: "+MPL,
        "MIT License: в опубликованном тексте нет отдельного явно сформулированного патентного гранта или условия прекращения патентных прав вследствие патентного иска. MIT не имеет нумерованных пунктов: предоставление разрешений содержится в первом абзаце, далее следуют сохранение уведомлений и отказ от гарантий. Отсутствие прямой оговорки не доказывает отсутствие подразумеваемых патентных прав. Официальный текст: "+MIT,
        "Ключевое различие: MPL 2.0 содержит прямо обозначенный патентный грант (2.1(b)) и условия прекращения (5.1 и 5.2); MIT License не содержит аналогичных прямых оговорок."
    ].join("\n\n");
    return {success:true,source:"bounded-patent-comparison",text};
}
