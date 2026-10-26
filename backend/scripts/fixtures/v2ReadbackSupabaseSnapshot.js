// Snapshot of one REAL Supabase RPC v4 transaction, rolled back after reading.
// No production Skill payload was modified or disclosed.
export const skillId="jessica-readback-v2-isolated-6ca6d9f7";
export const version=1;
export const facade={
  id:skillId,version,name:"Transient two-part Skill",enabled:false,status:"draft",
  moduleManifest:{
    version:2,
    counts:{examples:1,workflow:2},
    sha256:{
      examples:["84a95c8788b9ed5a3d169eaff9e9a20123db14cd8deb89db0efc2a0ca3e09ef3"],
      workflow:[
        "f2803d85dc3bd5fc5f73176b99adf1eb30f30ef0413d092de16a66a06318a6fd",
        "a8cb37c233decc7240c065304bdc5c65135f7a455240dc615b8f4aa65d74a851"
      ]
    }
  }
};
export const rows=[
  {module_key:"examples",part_no:0,
   data:[{task:"checksum roundtrip",status:"ok"}]},
  {module_key:"workflow",part_no:0,data:["read source"]},
  {module_key:"workflow",part_no:1,data:["verify output"]}
];
