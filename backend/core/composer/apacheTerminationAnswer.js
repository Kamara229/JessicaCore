/** Small source-scoped explanation of a patent termination condition. */
export function explainTermination(task,section) {
  if (!/(?:прекращ|прекрат|иск|terminate|litigation)/iu.test(String(task||"")))
    return null;
  const s=String(section||"").toLowerCase();
  const checks=["cross-claim","counterclaim","contribution incorporated",
    "direct or contributory","shall terminate","litigation is filed"];
  if (!checks.every(x=>s.includes(x)))return null;
  return "Если получатель Apache License 2.0 подаёт патентный иск (включая встречный или перекрёстный), утверждая, что произведение или включённый в него вклад нарушает патент, предоставленные по лицензии патентные права на это произведение прекращаются с даты подачи иска, а не с даты решения суда.";
}
