import {createHash} from "node:crypto";
import {materializeVerifiedDocumentAnalysisStage} from "./documentAnalysisStageMaterialization.js";
import {verifyDocumentStageWindowAnswer} from "./documentStageWindowVerification.js";
import {markDocumentQuotaInterruption} from "./documentQuotaCoverage.js";
import {documentStageTaskSha256} from "./documentStageTaskIdentity.js";

const sha=text=>createHash("sha256").update(text,"utf8").digest("hex");

/** Exactly one source-verified stage, not an analysis of the full document. */
export async function analyzeVerifiedDocumentStage(
    {task,source,verified,stageNumber,name},generate
) {
    if(typeof task!=="string"||!task.trim()||typeof generate!=="function"||
       typeof name!=="string"||!name||name.length>100)
        return {success:false,mode:"document-stage",stage:"document-stage-input"};
    const stage=materializeVerifiedDocumentAnalysisStage(source,verified,stageNumber);
    if(!stage)return {success:false,mode:"document-stage",
        stage:"document-stage-integrity",text:"Этап не прошёл проверку исходного текста."};
    const taskSha256=documentStageTaskSha256(task);
    if(!taskSha256)return {success:false,mode:"document-stage",
        stage:"document-stage-input"};
    const receipts=[],answers=[];
    const coverage=()=>({
        sourceSha256:stage.sourceSha256,sourceChars:stage.sourceChars,
        taskSha256,
        stageNumber,stageCount:stage.stageCount,
        stageSha256:stage.stageSha256,
        plannedWindows:stage.windowCount,processedWindows:receipts.length,
        totalSourceWindows:verified.totalWindows,
        verifiedWindows:receipts.map(x=>({...x})),
        stageAnalysisComplete:false,fullSourceAnalysisConfirmed:false,
        semanticCorrectnessConfirmed:false,resumeAvailable:false
    });
    for(const window of stage.windows){
        if(sha(window.text)!==window.windowSha256)
            return {success:false,mode:"document-stage",stage:"document-stage-integrity",
                readingCoverage:coverage()};
        const prompt=[
            "ЗАДАНИЕ:",task.trim().slice(0,3000),
            "ОДНО ОКНО ПРОВЕРЕННОГО ЭТАПА:",
            "Этап "+stageNumber+"/"+stage.stageCount+", окно "+window.number+
              ", раздел «"+window.title+"», символы "+window.startOffset+"–"+window.endOffset+".",
            "Цитируй фрагменты Ф1:Ф… только из переданного текста.",
            "Не утверждай, что проверены другие окна или весь документ."
        ].join("\n");
        const excerpt={name,text:window.text,truncated:false};
        let raw;
        try {
            raw=await generate(prompt,task,[excerpt]);
        }catch(error){
            if(error?.status===429)
                throw markDocumentQuotaInterruption(error,coverage(),window.number);
            return {success:false,mode:"document-stage",stage:"document-stage-provider-error",
                readingCoverage:coverage()};
        }
        const checked=verifyDocumentStageWindowAnswer(window,name,raw,task);
        if(!checked)return {success:false,mode:"document-stage",
            stage:"document-stage-window-evidence",
            text:"Ответ не прошёл проверку цитат; этап не завершён.",
            readingCoverage:coverage()};
        receipts.push(checked.receipt);
        answers.push("Окно "+window.number+" ("+window.startOffset+"–"+
            window.endOffset+"):\n"+checked.text);
    }
    return {success:true,mode:"document-stage",stage:"document-stage-verified",
        readingCoverage:{...coverage(),stageAnalysisComplete:true},
        stageReport:{sourceSha256:stage.sourceSha256,
            taskSha256,
            stageNumber,stageCount:stage.stageCount,stageSha256:stage.stageSha256,
            verifiedWindows:receipts.length,stageAnalysisComplete:true,
            fullSourceAnalysisConfirmed:false,semanticCorrectnessConfirmed:false},
        text:"Проверенный этап "+stageNumber+" из "+stage.stageCount+".\n\n"+
            answers.join("\n\n")+
            "\n\nПроверены цитаты этапа. Полнота всего документа и достоверность выводов не подтверждены."};
}
