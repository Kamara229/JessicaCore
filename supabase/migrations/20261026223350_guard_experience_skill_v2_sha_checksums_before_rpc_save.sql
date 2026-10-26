-- Audited SQL state for v2 Experience Skill checksum validation.
-- Original database migration: 20261026223350; corrected: 20261026223416.
-- Module functions remain small and independently testable.
-- Existing v1 records remain readable and write-compatible.

CREATE OR REPLACE FUNCTION public.jessica_experience_stable_json_v2(p_value jsonb)
 RETURNS text
 LANGUAGE plpgsql
 IMMUTABLE STRICT
 SET search_path TO ''
AS $function$
DECLARE
 v_text text;
BEGIN
 CASE jsonb_typeof(p_value)
 WHEN 'array' THEN
   SELECT '[' || coalesce(string_agg(public.jessica_experience_stable_json_v2(x.value),',' ORDER BY x.idx),'') || ']'
   INTO v_text FROM jsonb_array_elements(p_value) WITH ORDINALITY AS x(value,idx);
 WHEN 'object' THEN
   SELECT '{' || coalesce(string_agg(to_jsonb(x.key)::text||':'||public.jessica_experience_stable_json_v2(x.value),',' ORDER BY x.key COLLATE "C"),'') || '}'
   INTO v_text FROM jsonb_each(p_value) AS x(key,value);
 ELSE
   v_text:=p_value::text;
 END CASE;
 RETURN v_text;
END
$function$
;

CREATE OR REPLACE FUNCTION public.jessica_experience_v2_checksum_guard(p_payload jsonb, p_modules jsonb)
 RETURNS void
 LANGUAGE plpgsql
 SET search_path TO ''
AS $function$
DECLARE
 manifest jsonb;
 counts jsonb;
 hashes jsonb;
 name text;
 n int;
 idx int;
 expected text;
 actual text;
 part_data jsonb;
 matching_parts int;
BEGIN
 manifest:=p_payload->'moduleManifest';
 IF manifest->'version'='1'::jsonb THEN RETURN; END IF;
 IF manifest->'version' IS DISTINCT FROM '2'::jsonb THEN
   RAISE EXCEPTION 'Skill module manifest version must be 1 or 2';
 END IF;
 counts:=manifest->'counts';
 hashes:=manifest->'sha256';
 IF jsonb_typeof(counts)<>'object' OR jsonb_typeof(hashes)<>'object' THEN
   RAISE EXCEPTION 'Skill v2 checksum manifest missing';
 END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(hashes)) <>
    (SELECT count(*) FROM jsonb_object_keys(counts)) THEN
   RAISE EXCEPTION 'Skill v2 checksum keys do not match module keys';
 END IF;
 FOR name,n IN SELECT key,value::integer FROM jsonb_each_text(counts) LOOP
   IF n<1 OR n>100 OR jsonb_typeof(hashes->name)<>'array'
      OR jsonb_array_length(hashes->name)<>n THEN
     RAISE EXCEPTION 'Skill v2 checksum count mismatch for %',name;
   END IF;
   FOR idx IN 0..(n-1) LOOP
     SELECT count(*) INTO matching_parts
       FROM jsonb_array_elements(p_modules) AS x
       WHERE x->>'moduleKey'=name AND x->>'part'=idx::text;
     SELECT x->'data' INTO part_data
       FROM jsonb_array_elements(p_modules) AS x
       WHERE x->>'moduleKey'=name AND x->>'part'=idx::text
       LIMIT 1;
     IF matching_parts<>1 OR jsonb_typeof(part_data)<>'array'
       OR jsonb_array_length(part_data)=0 THEN
       RAISE EXCEPTION 'Skill v2 checksum part missing or duplicated for % part %',name,idx;
     END IF;
     expected:=hashes->name->>idx;
     IF expected IS NULL OR expected!~'^[0-9a-f]{64}$' THEN
       RAISE EXCEPTION 'Skill v2 checksum invalid for % part %',name,idx;
     END IF;
     actual:=encode(extensions.digest(public.jessica_experience_stable_json_v2(part_data),'sha256'),'hex');
     IF actual<>expected THEN
       RAISE EXCEPTION 'Skill v2 checksum mismatch for % part %',name,idx;
     END IF;
   END LOOP;
 END LOOP;
END
$function$
;

CREATE OR REPLACE FUNCTION public.save_jessica_experience_skill_v4(p_skill_id text, p_version integer, p_previous_version integer, p_mode text, p_enabled boolean, p_payload jsonb, p_metadata jsonb, p_evidence jsonb DEFAULT '[]'::jsonb, p_modules jsonb DEFAULT '[]'::jsonb)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO ''
AS $function$
declare
 result jsonb;
 item jsonb;
 k text;
 n integer;
 part integer;
 total integer := 0;
 manifest jsonb;
 expected jsonb;
 actual integer;
begin
 perform public.jessica_experience_v2_checksum_guard(p_payload,p_modules);
 if jsonb_typeof(p_payload) <> 'object' then
   raise exception 'Skill facade must be a JSON object';
 end if;
 if jsonb_typeof(p_modules) <> 'array' or jsonb_array_length(p_modules) > 100 then
   raise exception 'Skill modules must be a bounded JSON array';
 end if;
 manifest := p_payload->'moduleManifest';
 expected := manifest->'counts';
 if jsonb_typeof(expected) <> 'object' then
   raise exception 'Versioned skill module manifest is missing';
 end if;
 if (select count(*) from jsonb_object_keys(expected)) < 1 then
   raise exception 'Skill module manifest has no modules';
 end if;
 for item in select value from jsonb_array_elements(p_modules) loop
   k := item->>'moduleKey';
   if k is null or k not in
    ('workflow','validationRules','constraints','strategy','sourcePriority','successfulPatterns','failurePatterns','avoidPatterns','examples') then
     raise exception 'Unknown skill module';
   end if;
   if item->>'part' !~ '^([0-9]|[1-9][0-9]{1,2})$' then
     raise exception 'Invalid module part number';
   end if;
   part := (item->>'part')::integer;
   if jsonb_typeof(item->'data') <> 'array' or
     octet_length((item->'data')::text) > 6144 then
     raise exception 'Skill module must be an array <= 6144 bytes';
   end if;
   total:= total+1;
 end loop;
 if total=0 or total <> (select sum((value::text)::integer) from jsonb_each_text(expected)) then
   raise exception 'Module manifest count mismatch';
 end if;
 for k,n in select key, value::integer from jsonb_each_text(expected) loop
   select count(*) into actual from jsonb_array_elements(p_modules) x
    where x->>'moduleKey'=k;
   if actual<>n then raise exception 'Module type count mismatch'; end if;
   if n<1 or n>100 then raise exception 'Bad module count'; end if;
 end loop;
 result:=public.save_jessica_experience_skill_v3(
     p_skill_id,p_version,p_previous_version,p_mode,p_enabled,p_payload,p_metadata,p_evidence);
 if result->>'success' is distinct from 'true' or result->>'existing'='true' then
   return result || jsonb_build_object('writerVersion','v4-modules');
 end if;
 for item in select value from jsonb_array_elements(p_modules) loop
   insert into public.jessica_experience_skill_modules(
    skill_id,version,module_key,part_no,data)
   values(p_skill_id,p_version,item->>'moduleKey',(item->>'part')::integer,item->'data');
 end loop;
 return result || jsonb_build_object('moduleCount',total,'writerVersion','v4-modules');
end;
$function$
;
