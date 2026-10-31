export function solveThreeContainerLinear(p){
 if(!p || !Number.isInteger(p.multiple) || p.multiple<2 || p.multiple>20 ||
    !Number.isInteger(p.offset) || p.offset<0 || p.offset>9999 ||
    ![-1,1].includes(p.offsetSign) ||
    !Number.isInteger(p.sum) || p.sum<0 || p.sum>999999)return null;
 const divisor=2*p.multiple+1;
 const numerator=p.sum-p.offsetSign*p.offset;
 const integer=numerator>=0&&numerator%divisor===0;
 const second=integer?numerator/divisor:null;
 const first=integer?p.multiple*second:null;
 const third=integer?first+p.offsetSign*p.offset:null;
 const feasible=integer&&third>=0&&first+second+third===p.sum;
 return {divisor,numerator,feasible,first:feasible?first:null,
   second:feasible?second:null,third:feasible?third:null};
}
