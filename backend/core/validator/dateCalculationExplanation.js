/** Shared non-AI completeness check for requested date-difference explanations. */
export function dateCalculationExplanationMissing(task,answer){
  const request=String(task||"");
  if(!/(дн|day)/iu.test(request))return false;
  const explicit=/(?:покажи|приведи|выведи)\s+(?:сам\s+|подробный\s+|полный\s+)?расч[её]т|show\s+(?:the\s+)?calculation/iu.test(request);
  const requested=explicit||/(объясн|поясн|explain)/iu.test(request);
  if(!requested)return false;
  const response=String(answer||"");
  if(explicit)return !/Расч[её]т\s*:/iu.test(response);
  return !/(начальная дата|Расч[её]т\s*:|учитываем|counted dates)/iu.test(response);
}
