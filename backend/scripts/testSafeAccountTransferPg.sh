#!/usr/bin/env bash
set -euo pipefail
export PGHOST=localhost PGPORT=5432 PGUSER=transfer PGPASSWORD=transfer PGDATABASE=transfer
psql -X -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE accounts(account_id integer PRIMARY KEY, balance numeric NOT NULL CHECK (balance >= 0));
INSERT INTO accounts VALUES (1,100),(2,100),(3,100),(4,100);
SQL
node --input-type=module -e 'import {safeAccountTransferSqlTool} from "./backend/tools/safeAccountTransferSqlTool.js"; process.stdout.write(safeAccountTransferSqlTool().data.sql)' | psql -X -v ON_ERROR_STOP=1
(psql -X -v ON_ERROR_STOP=1 -c 'SELECT transfer_funds(1,2,100)' >/tmp/transfer-a.log 2>&1 && echo pass >/tmp/transfer-a.status || echo fail >/tmp/transfer-a.status) &
(psql -X -v ON_ERROR_STOP=1 -c 'SELECT transfer_funds(1,2,100)' >/tmp/transfer-b.log 2>&1 && echo pass >/tmp/transfer-b.status || echo fail >/tmp/transfer-b.status) &
wait
test "$(cat /tmp/transfer-a.status)$(cat /tmp/transfer-b.status)" = passfail -o "$(cat /tmp/transfer-a.status)$(cat /tmp/transfer-b.status)" = failpass
test "$(psql -X -At -c 'SELECT balance FROM accounts WHERE account_id=1')" = '0'
test "$(psql -X -At -c 'SELECT balance FROM accounts WHERE account_id=2')" = '200'
(psql -X -v ON_ERROR_STOP=1 -c 'SELECT transfer_funds(3,4,10)' >/tmp/reverse-a.log 2>&1 && echo pass >/tmp/reverse-a.status || echo fail >/tmp/reverse-a.status) &
(psql -X -v ON_ERROR_STOP=1 -c 'SELECT transfer_funds(4,3,10)' >/tmp/reverse-b.log 2>&1 && echo pass >/tmp/reverse-b.status || echo fail >/tmp/reverse-b.status) &
wait
test "$(cat /tmp/reverse-a.status)$(cat /tmp/reverse-b.status)" = passpass
test "$(psql -X -At -c 'SELECT balance FROM accounts WHERE account_id=3')" = '100'
test "$(psql -X -At -c 'SELECT balance FROM accounts WHERE account_id=4')" = '100'
for statement in 'SELECT transfer_funds(1,999,10)' 'SELECT transfer_funds(3,4,0)' 'SELECT transfer_funds(3,3,10)' 'SELECT transfer_funds(1,2,10)'; do
  if psql -X -v ON_ERROR_STOP=1 -c "$statement" >/tmp/bad-transfer.log 2>&1; then
    echo "Invalid transfer unexpectedly succeeded: $statement"; exit 1
  fi
done
psql -X -v ON_ERROR_STOP=1 -c 'UPDATE accounts SET balance=100 WHERE account_id=1'
psql -X -v ON_ERROR_STOP=1 <<'SQL'
CREATE FUNCTION reject_credit() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.account_id=2 AND NEW.balance>OLD.balance THEN RAISE EXCEPTION 'credit rejected'; END IF;
  RETURN NEW;
END;
$$;
CREATE TRIGGER reject_credit_t BEFORE UPDATE ON accounts FOR EACH ROW EXECUTE FUNCTION reject_credit();
SQL
if psql -X -v ON_ERROR_STOP=1 -c 'SELECT transfer_funds(1,2,10)' >/tmp/rollback.log 2>&1; then echo 'Credit failure not detected'; exit 1; fi
test "$(psql -X -At -c 'SELECT balance FROM accounts WHERE account_id=1')" = '100'
test "$(psql -X -At -c 'SELECT balance FROM accounts WHERE account_id=2')" = '200'
echo 'PostgreSQL transfer concurrency and rollback checks passed'
