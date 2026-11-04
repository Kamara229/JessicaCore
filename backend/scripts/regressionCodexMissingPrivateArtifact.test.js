import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {missingCalibriArtifactHandoff} from "../core/codex/relayMissingCalibriArtifact.js";

const id="5d3a3d96-ee9c-412b-8d85-81c66707f963";
const body="Проверь 27 ответов v108 вручную: FP32 — 4/6 верных, CB59 и v102 — 0/6. Если исходные ответы недоступны, сообщи об этом. Не подтверждай результаты без фактической проверки.";
const envelope="JESSICA_REPLY_"+id+"\n"+body+"\nJESSICA_END_"+id;

test("exact previously failed Codex turn needs no AI provider, model test, or private-data hallucination",async()=>{
    for(const message of [body,envelope]){
        let calls=0;
        const out=await interpretRelayTurn({message,plan:"CALIBRI: анализ v108"},async()=>{
            calls++;
            throw Error("groq 429 or invalid json");
        });
        assert.equal(calls,0);
        assert.equal(out.action,"reply");
        assert.equal(out.task,"");
        assert.equal(out.continueThread,false);
        assert.match(out.reply,/не подтверждаю оценки/);
        assert.match(out.reply,/не запускала/);
        assert.match(out.reply,/GitHub Actions/);
    }
});

test("malformed markers, actual source contents or different tasks cannot falsely imply missing access",()=>{
    assert.equal(missingCalibriArtifactHandoff("JESSICA_REPLY_"+id+"\n"+body),null);
    assert.equal(missingCalibriArtifactHandoff("JESSICA_REPLY_"+id+"\n"+body+"\nJESSICA_END_00000000-0000-0000-0000-000000000000"),null);
    assert.equal(missingCalibriArtifactHandoff(body.replace("27 ответов","28 ответов")),null);
    assert.equal(missingCalibriArtifactHandoff(body.replace("Если исходные ответы недоступны, сообщи об этом.","")),null);
    assert.equal(missingCalibriArtifactHandoff(body+"\nНовый запрос"),null);
    assert.equal(missingCalibriArtifactHandoff(body+" https://github.com/some/source"),null);
    assert.equal(missingCalibriArtifactHandoff("Запусти тест Jessica по задаче: Сколько будет 2+2?"),null);
});

test("missing-source status handoff never manufactures a verified result",()=>{
    const r=missingCalibriArtifactHandoff(body);
    assert.equal(r.action,"reply");
    assert.doesNotMatch(r.reply,/\b(?:PASS|SUCCESS|подтверждено|выполнен тест)\b/iu);
    assert.ok(!r.reply.includes("18/18"));
});
