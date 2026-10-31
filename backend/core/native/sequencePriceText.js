const money=n=>Number(n.toFixed(2)).toLocaleString("ru-RU");
export function formatSequencePrice(r){
 if(!r)return "";
 const factorUp=(100+r.up)/100,factorDown=(100-r.down)/100;
 return "Первоначальная цена: "+money(r.base)+" ₽. "+
 "После повышения на "+r.up+"%: "+money(r.base)+" × "+factorUp+" = "+money(r.afterRaise)+" ₽. "+
 "После снижения на "+r.down+"%: "+money(r.afterRaise)+" × "+factorDown+" = "+money(r.final)+" ₽. "+
 "Изменение: "+money(r.final)+" − "+money(r.base)+" = "+money(r.final-r.base)+" ₽; "+
 "("+money(r.final)+" − "+money(r.base)+") / "+money(r.base)+" × 100% = "+
 money(r.changePercent)+"%. Ответ: "+money(r.final)+" ₽, изменение "+money(r.changePercent)+"%.";
}
