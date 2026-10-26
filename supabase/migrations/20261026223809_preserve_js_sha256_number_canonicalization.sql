-- Preserve ECMAScript JSON.stringify numeric formatting for v2 module hashes.
-- This migration follows 20261026223350 and database correction 20261026223416.
-- Keeps checksum verification compatible with 1e-7, 1e+21 and ordinary decimals.

CREATE OR REPLACE FUNCTION public.jessica_experience_json_number_v2(p_value jsonb)
 RETURNS text
 LANGUAGE plpgsql
 IMMUTABLE STRICT
 SET search_path TO ''
AS $function$
DECLARE
 s text;
 sign_value text:='';
 whole text;
 fraction text;
 digits text;
 magnitude numeric;
 exponent_value int;
 leading_digit int;
 mantissa text;
BEGIN
 IF jsonb_typeof(p_value)<>'number' THEN
   RAISE EXCEPTION 'Expected JSON number';
 END IF;
 s:=p_value::text;
 IF s LIKE '-%' THEN
   sign_value:='-';
   s:=substr(s,2);
 END IF;
 IF position('.' in s)>0 THEN
   s:=rtrim(rtrim(s,'0'),'.');
 END IF;
 IF s='' OR s='0' THEN RETURN '0'; END IF;
 magnitude:=s::numeric;
 IF magnitude>=1000000000000000000000::numeric
   OR magnitude<0.000001::numeric THEN
   whole:=split_part(s,'.',1);
   fraction:=split_part(s,'.',2);
   IF whole<>'0' THEN
     exponent_value:=length(whole)-1;
     digits:=whole||fraction;
   ELSE
     leading_digit:=length(substring(fraction from '^[0]*'))+1;
     exponent_value:=-leading_digit;
     digits:=substr(fraction,leading_digit);
   END IF;
   digits:=rtrim(digits,'0');
   mantissa:=left(digits,1);
   IF length(digits)>1 THEN mantissa:=mantissa||'.'||substr(digits,2); END IF;
   RETURN sign_value||mantissa||'e'||
     CASE WHEN exponent_value>=0 THEN '+' ELSE '' END||exponent_value::text;
 END IF;
 RETURN sign_value||s;
END
$function$
;

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
 WHEN 'number' THEN
   v_text:=public.jessica_experience_json_number_v2(p_value);
 ELSE
   v_text:=p_value::text;
 END CASE;
 RETURN v_text;
END
$function$
;
