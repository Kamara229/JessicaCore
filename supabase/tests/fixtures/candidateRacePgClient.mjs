import {spawn} from "node:child_process";
import {setTimeout as delay} from "node:timers/promises";

// Isolated CI PostgreSQL only. No application credentials or production writes.
export function assertIsolatedCi(){
  if (process.env.JESSICA_ISOLATED_PG !== "true" ||
      process.env.PGHOST !== "127.0.0.1" ||
      process.env.PGDATABASE !== "jessica_ci") {
    throw Error("Candidate race regression requires isolated CI PostgreSQL");
  }
}

export function sql(query, app="jessica-ci-observer"){
  return new Promise((resolve,reject)=>{
    const child=spawn("psql",["-X","-A","-t","-q","-v","ON_ERROR_STOP=1","-f","-"],{
      env:{...process.env,PGAPPNAME:app},stdio:["pipe","pipe","pipe"]
    });
    let stdout="",stderr="";
    child.stdout.on("data",data=>{stdout+=data;});
    child.stderr.on("data",data=>{stderr+=data;});
    child.on("error",reject);
    child.on("close",code=>resolve({code,stdout:stdout.trim(),stderr:stderr.trim()}));
    child.stdin.end(query);
  });
}

export async function waitFor(sqlQuery,timeoutMs=3500){
  const end=Date.now()+timeoutMs;
  while(Date.now()<end){
    const result=await sql(sqlQuery);
    if(result.code!==0)throw Error(result.stderr);
    if(result.stdout==="1")return true;
    await delay(70);
  }
  return false;
}
