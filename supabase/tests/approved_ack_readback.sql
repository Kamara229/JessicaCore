-- Physical PostgreSQL guard for a final APPROVED write whose client ACK
-- was lost. The nested subtransaction unconditionally rolls back test rows.
DO $ack_probe$
DECLARE
  v_id text := 'proposal-final-ack-probe-'||gen_random_uuid()::text;
  v_saved_at timestamptz;
  v_after_at timestamptz;
  v_count integer;
BEGIN
  BEGIN
    INSERT INTO public.learning_proposals(id,source,action,status,processing_at)
    VALUES(v_id,'isolated-final-ack-readback','NEW_SKILL','PROCESSING',now());
    -- Server commits the intended final decision (a missing network ACK
    -- cannot roll a committed transaction back).
    UPDATE public.learning_proposals
      SET status='APPROVED',processing_at=null,approved_at=now()
      WHERE id=v_id AND status='PROCESSING';
    GET DIAGNOSTICS v_count=ROW_COUNT;
    IF v_count<>1 THEN RAISE EXCEPTION 'Initial Approval did not commit'; END IF;

    -- The client lost its update response; durable status readback wins.
    SELECT approved_at INTO v_saved_at
      FROM public.learning_proposals WHERE id=v_id AND status='APPROVED';
    IF v_saved_at IS NULL THEN
      RAISE EXCEPTION 'Final Approval ACK could not be recovered';
    END IF;
    -- No second write on repeat confirmation; timestamp is immutable.
    UPDATE public.learning_proposals
       SET status='APPROVED',approved_at=now()
       WHERE id=v_id AND status='PROCESSING';
    GET DIAGNOSTICS v_count=ROW_COUNT;
    IF v_count<>0 THEN
      RAISE EXCEPTION 'Repeating status ACK changed terminal Proposal';
    END IF;
    SELECT approved_at INTO v_after_at FROM public.learning_proposals WHERE id=v_id;
    IF v_after_at IS DISTINCT FROM v_saved_at THEN
      RAISE EXCEPTION 'Terminal status timestamp changed after replay';
    END IF;
    -- Neither stale recovery nor a competing rejection can reopen it.
    UPDATE public.learning_proposals
       SET status='PENDING_APPROVAL',approved_at=null
       WHERE id=v_id AND status='PROCESSING';
    GET DIAGNOSTICS v_count=ROW_COUNT;
    IF v_count<>0 THEN RAISE EXCEPTION 'Approved Proposal requeued'; END IF;
    UPDATE public.learning_proposals
       SET status='REJECTED',approved_at=null,rejected_at=now()
       WHERE id=v_id AND status='PROCESSING';
    GET DIAGNOSTICS v_count=ROW_COUNT;
    IF v_count<>0 THEN RAISE EXCEPTION 'Approved Proposal overwritten'; END IF;
    IF NOT EXISTS (
      SELECT 1 FROM public.learning_proposals
       WHERE id=v_id AND status='APPROVED' AND approved_at=v_saved_at
         AND processing_at IS NULL
    ) THEN RAISE EXCEPTION 'Lost-ACK durable final status proof failed'; END IF;
    RAISE EXCEPTION 'ROLLBACK_FINAL_ACK_PROBE' USING ERRCODE='ZX001';
  EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
  END;
END
$ack_probe$;