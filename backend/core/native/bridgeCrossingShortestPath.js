/**
 * Enumerate all legal flashlight states; Dijkstra certifies the global minimum.
 * The mask identifies people on the destination bank; side=1 means torch there.
 */
export function shortestBridgeCrossing(times) {
  if(!Array.isArray(times) || times.length!==4 ||
     times.some(t=>!Number.isSafeInteger(t)||t<=0||t>120))return null;
  const n=times.length, size=(1<<n)*2, goal=((1<<n)-1)*2+1;
  const dist=Array(size).fill(Infinity), seen=Array(size).fill(false), prev=Array(size).fill(null);
  dist[0]=0;
  for(let pass=0;pass<size;pass++){
    let u=-1;
    for(let k=0;k<size;k++)if(!seen[k]&&(u<0||dist[k]<dist[u]))u=k;
    if(u<0||!Number.isFinite(dist[u]))break;
    seen[u]=true;
    const mask=u>>1, side=u&1, eligible=[];
    for(let i=0;i<n;i++)if(((mask>>i)&1)===side)eligible.push(i);
    for(let a=0;a<eligible.length;a++)for(let b=a;b<eligible.length;b++){
      if(a===b && b>a)continue;
      const people=a===b?[eligible[a]]:[eligible[a],eligible[b]];
      const bits=people.reduce((v,i)=>v|(1<<i),0);
      const nextMask=side===0?mask|bits:mask&~bits;
      const v=nextMask*2+(1-side);
      const cost=Math.max(...people.map(i=>times[i]));
      if(dist[u]+cost<dist[v]){
        dist[v]=dist[u]+cost;
        prev[v]={from:u,people,fromSide:side,cost};
      }
    }
  }
  if(!Number.isFinite(dist[goal]))return null;
  const route=[];
  for(let state=goal;state!==0;){
    const step=prev[state];if(!step)return null;
    route.push({people:step.people,fromSide:step.fromSide,cost:step.cost});
    state=step.from;
  }
  route.reverse();
  return {minutes:dist[goal],route,distances:dist.map(v=>Number.isFinite(v)?v:null),
    reachable:dist.filter(Number.isFinite).length};
}
