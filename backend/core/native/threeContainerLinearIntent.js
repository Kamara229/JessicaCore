export function parseThreeContainerLinearTask(task){
 const t=String(task||"");
 if(!/(тр[её]х|3) корзин/iu.test(t))return null;
 const a=t.match(/первой (вдвое|втрое|вчетверо|в ?\d+ раз[а]?) больше,? чем во второй/iu);
 const b=t.match(/третьей на (\d+) (меньше|больше),? чем в первой/iu);
 const s=t.match(/всего (\d+)/iu);
 if(!a||!b||!s)return null;
 const mult={вдвое:2,втрое:3,вчетверо:4}[a[1].toLowerCase()]||Number(a[1].match(/\d+/)?.[0]);
 if(!Number.isInteger(mult)||mult<2||mult>20)return null;
 return {multiple:mult,offset:Number(b[1]),offsetSign:/меньше/iu.test(b[2])?-1:1,sum:Number(s[1])};
}
