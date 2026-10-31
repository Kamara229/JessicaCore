export function formatThreeContainerLinear(p,r){
 if(!p||!r)return "";
 const sign=p.offsetSign<0?"−":"+";
 const expr=p.multiple+"x "+sign+" "+p.offset;
 const eq=p.multiple+"x + x + ("+expr+") = "+p.sum;
 const derive="Пусть во второй корзине x. В первой — "+p.multiple+
    "x; в третьей — "+expr+". Уравнение: "+eq+
    "; "+r.divisor+"x = "+r.numerator+". ";
 if(!r.feasible)return derive+"Не существует решения в целых неотрицательных числах.";
 return derive+"Значит x = "+r.second+". Ответ: первая — "+r.first+
    ", вторая — "+r.second+", третья — "+r.third+
    ". Проверка: "+r.first+" = "+p.multiple+" × "+r.second+
    "; "+r.third+" = "+r.first+" "+sign+" "+p.offset+
    "; "+r.first+" + "+r.second+" + "+r.third+" = "+p.sum+".";
}
