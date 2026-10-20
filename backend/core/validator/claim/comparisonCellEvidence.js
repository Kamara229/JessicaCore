import {matchesAttributedEvidence} from "./claimEvidenceMatcher.js";
import {isUnsupportedNegativeComparisonClaim} from "./negativeComparisonClaim.js";
import {unsupportedComparisonDuty} from "./comparisonDutyEvidence.js";
import {unsupportedComparisonClaimTopic} from "./comparisonClaimRelevance.js";
import {rejectsUnsupportedLegalEffect} from "./comparisonLegalEffectGate.js";
import {unsupportedAbsolutePermission} from "./comparisonAbsolutePermissionGuard.js";
import {contradictsExpressCopyrightGrant} from "./comparisonGrantPolarityGate.js";
import {unsupportedEndorsementClaim} from "./comparisonEndorsementEvidence.js";

/* Each supplied quote must be verbatim, source-attributed, and relevant.
 * Several independent facts in one cell require several supporting clauses.
 * Passing these gates is not a substitute for semantic fact verification. */
function compoundClauseMissing(statement,evidence) {
  const text=String(statement||"").toLowerCase();
  const proof=String(evidence||"").toLowerCase();
  if(/патент|patent/.test(text)){
    if(/прекращ|растор|terminat|отзыв/.test(text)&&!/terminat|cease|revok/.test(proof))return true;
    if(/грант|предостав|выда[её]|grant/.test(text)&&!/grant|licensable by such contributor/.test(proof))return true;
  }
  if(/уведомлен|copyright|копирайт|авторск/.test(text)
     &&/исходн|source/.test(text)&&/бинарн|binary/.test(text)
     &&(!/source code|source form/.test(proof)||!/binary form|binary code/.test(proof)))
    return true;
  return false;
}
export function verifyComparisonCellEvidence(cell,quotes,originalSources) {
  const statement=cell.criterion+": "+cell.value;
  const claim={claim:statement,sourceUrl:cell.sourceUrl,
    evidence:quotes.join(" / "),supported:false,reason:""};
  if(!quotes.length||quotes.some(q=>q.length<18||q.split(/\s+/).length<4)){
    claim.reason="Не хватает содержательных цитат";return claim;
  }
  for(const quote of quotes) {
    if(!matchesAttributedEvidence({sourceUrl:cell.sourceUrl,evidence:quote},originalSources)){
      claim.reason="Одна или несколько цитат не найдены в указанном первоисточнике";
      return claim;
    }
  }
  const proof=quotes.join(" ");
  if(isUnsupportedNegativeComparisonClaim({claim:statement,evidence:proof,sourceUrl:cell.sourceUrl},true)){
    claim.reason="Отрицательный вывод не доказан";
  } else if(unsupportedComparisonDuty(statement,proof)){
    claim.reason="Цитаты не доказывают обязательное требование";
  } else if(unsupportedAbsolutePermission(cell,quotes)){
    claim.reason="Абсолютное разрешение не подтверждено текстом лицензии";
  } else if(contradictsExpressCopyrightGrant(cell,quotes)){
    claim.reason="Утверждение об отзыве авторской лицензии противоречит первоисточнику";
  } else if(unsupportedEndorsementClaim(cell.value,quotes)){
    claim.reason="Цитата не подтверждает отдельный запрет заявлять об официальном одобрении";
  } else if(rejectsUnsupportedLegalEffect(cell,quotes)){
    claim.reason="Цитата не подтверждает правовой смысл утверждения";
  } else if(unsupportedComparisonClaimTopic(statement,proof)||compoundClauseMissing(statement,proof)){
    claim.reason="Цитаты не подтверждают все части утверждения";
  } else{
    claim.supported=true;
  }
  return claim;
}
