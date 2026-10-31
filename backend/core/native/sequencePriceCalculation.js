export function calculateSequentialPrice(input={}){
 const base=Number(input.price),up=Number(input.increase),down=Number(input.decrease);
 if(!Number.isFinite(base)||base<=0||base>100000000||!Number.isInteger(up)||up<0||up>100||!Number.isInteger(down)||down<0||down>100)return null;
 const afterRaise=base*(100+up)/100,final=afterRaise*(100-down)/100;
 const changePercent=(100+up)*(100-down)/100-100;
 return {base,up,down,afterRaise,final,changePercent};
}
