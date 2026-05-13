#!/usr/bin/env bash
# Curl-only end-to-end smoke test for the Modal HTTP endpoint on a luma plan.
#
# Mirrors the assertions in scripts/verify_modal_endpoint.py but using only
# curl + jq so it's easy to drop into CI / paste into a debugging session
# without any Python deps. Verifies:
#
#   1. /v1/health is 200 + correct service banner.
#   2. POST /v1/predict submits a luma task_suite and returns a run_id.
#   3. A second submit with the same profile_id returns 409 with the
#      conflicting run_id in the detail.
#   4. action=status polling returns a paused-or-terminal state within a
#      bounded window (the executor crashing on a missing brain dep is
#      still a "terminal" answer — we're testing the HTTP wire, not the
#      brain).
#   5. action=cancel releases the lock; a third submit with the same
#      profile_id then succeeds.
#
# Usage:
#   ENDPOINT=https://getmason--mantis-cua-server-api.modal.run \
#   MANTIS_API_TOKEN="..." \
#   ./scripts/verify_modal_luma_curl.sh
#
# All assertions are fatal — exit 0 only on full success.

set -euo pipefail

: "${ENDPOINT:=https://getmason--mantis-cua-server-api.modal.run}"
: "${MANTIS_API_TOKEN:?MANTIS_API_TOKEN must be set}"
: "${PLAN_PATH:=plans/luma-extract.json}"
: "${PROFILE_ID:=verify-luma-curl}"
: "${WORKFLOW_ID:=verify-luma-curl-v1}"
: "${MODEL:=claude}"

H_AUTH=(-H "X-Mantis-Token: ${MANTIS_API_TOKEN}")
H_JSON=(-H "Content-Type: application/json")

fail() { echo "❌ $*" >&2; exit 1; }
ok()   { echo "✅ $*"; }

# ── 1. health ────────────────────────────────────────────────────
echo "[1/5] GET ${ENDPOINT}/v1/health"
health_status=$(curl -fsSL -o /tmp/mantis_health.json -w '%{http_code}' "${ENDPOINT}/v1/health")
[[ "${health_status}" == "200" ]] || fail "health returned ${health_status}"
grep -q '"status":"ok"' /tmp/mantis_health.json || fail "health body missing status:ok"
grep -q '"service":"mantis-cua-modal-api"' /tmp/mantis_health.json || fail "health body missing service banner"
ok "health 200 — $(cat /tmp/mantis_health.json)"

# ── Build a task_suite from the on-disk luma plan ───────────────
[[ -f "${PLAN_PATH}" ]] || fail "plan file not found: ${PLAN_PATH} (run from repo root)"
SUITE=$(jq -n \
    --slurpfile plan "${PLAN_PATH}" \
    --arg profile "${PROFILE_ID}" \
    --arg workflow "${WORKFLOW_ID}" \
    '{
        session_name: "verify-luma-curl",
        _micro_plan: ($plan[0].steps // $plan[0]),
        _plan_signature: "luma-curl-fixed",
        _profile_id:  $profile,
        _workflow_id: $workflow,
        _state_key:   $workflow,
        _max_cost: 1.0,
        _max_time_minutes: 5,
        tasks: []
    }')

# ── 2. first submit returns 200 + run_id ────────────────────────
echo "[2/5] POST /v1/predict — first luma submit"
submit_body=$(jq -n \
    --argjson suite "${SUITE}" \
    --arg profile "${PROFILE_ID}" \
    --arg workflow "${WORKFLOW_ID}" \
    --arg model "${MODEL}" \
    '{
        detached: true,
        task_suite: $suite,
        profile_id: $profile,
        workflow_id: $workflow,
        cua_model: $model,
        max_steps: 4,
        max_cost: 1.0,
        max_time_minutes: 5
    }')

submit_resp=$(mktemp)
submit_status=$(curl -sS -o "${submit_resp}" -w '%{http_code}' \
    -X POST "${ENDPOINT}/v1/predict" "${H_AUTH[@]}" "${H_JSON[@]}" -d "${submit_body}")
[[ "${submit_status}" == "200" ]] || fail "first submit returned ${submit_status}: $(cat "${submit_resp}")"

run_id=$(jq -r '.run_id' "${submit_resp}")
[[ -n "${run_id}" && "${run_id}" != "null" ]] || fail "no run_id in response: $(cat "${submit_resp}")"
got_profile=$(jq -r '.payload.profile_id' "${submit_resp}")
got_workflow=$(jq -r '.payload.workflow_id' "${submit_resp}")
[[ "${got_profile}" == *"__${PROFILE_ID}"  ]] || fail "profile_id round-trip failed: ${got_profile}"
[[ "${got_workflow}" == *"__${WORKFLOW_ID}" ]] || fail "workflow_id round-trip failed: ${got_workflow}"
ok "first submit 200 — run_id=${run_id}, profile_id=${got_profile}"

# ── 3. duplicate profile_id → 409 ───────────────────────────────
echo "[3/5] POST /v1/predict — duplicate profile_id (expect 409)"
dup_resp=$(mktemp)
dup_status=$(curl -sS -o "${dup_resp}" -w '%{http_code}' \
    -X POST "${ENDPOINT}/v1/predict" "${H_AUTH[@]}" "${H_JSON[@]}" -d "${submit_body}")
[[ "${dup_status}" == "409" ]] || fail "expected 409, got ${dup_status}: $(cat "${dup_resp}")"
grep -q "${run_id}" "${dup_resp}" || fail "409 detail must surface the held run_id=${run_id}: $(cat "${dup_resp}")"
ok "duplicate 409 — held run_id surfaced in detail"

# ── 4. status poll until terminal-or-paused ─────────────────────
echo "[4/5] POST /v1/predict — action=status until non-running"
poll_body=$(jq -n --arg run_id "${run_id}" '{action: "status", run_id: $run_id}')
deadline=$(( $(date +%s) + 60 ))
final_status=""
while (( $(date +%s) < deadline )); do
    status_resp=$(curl -sS -X POST "${ENDPOINT}/v1/predict" \
        "${H_AUTH[@]}" "${H_JSON[@]}" -d "${poll_body}")
    final_status=$(jq -r '.status' <<< "${status_resp}")
    case "${final_status}" in
        queued|running) sleep 2; continue;;
        paused|succeeded|failed|cancelled) break;;
        *) fail "unexpected status: ${final_status} (body: ${status_resp})";;
    esac
done
[[ -n "${final_status}" && "${final_status}" != "queued" && "${final_status}" != "running" ]] \
    || fail "status never left running within 60s (last=${final_status})"
ok "status reached ${final_status} for run_id=${run_id}"

# ── 5. cancel + re-submit succeeds ──────────────────────────────
echo "[5/5] cancel + re-submit on same profile_id"
cancel_body=$(jq -n --arg run_id "${run_id}" '{action: "cancel", run_id: $run_id}')
cancel_resp=$(curl -sS -X POST "${ENDPOINT}/v1/predict" \
    "${H_AUTH[@]}" "${H_JSON[@]}" -d "${cancel_body}")
echo "  cancel: $(jq -r '.status' <<< "${cancel_resp}")"

# Brief beat for the volume commit on lock release.
sleep 2

resubmit_resp=$(mktemp)
resubmit_status=$(curl -sS -o "${resubmit_resp}" -w '%{http_code}' \
    -X POST "${ENDPOINT}/v1/predict" "${H_AUTH[@]}" "${H_JSON[@]}" -d "${submit_body}")
[[ "${resubmit_status}" == "200" ]] || fail "re-submit after cancel returned ${resubmit_status}: $(cat "${resubmit_resp}")"
new_run_id=$(jq -r '.run_id' "${resubmit_resp}")
[[ "${new_run_id}" != "${run_id}" ]] || fail "re-submit returned the same run_id (lock not released)"
ok "re-submit 200 — new run_id=${new_run_id} (lock released)"

# Tidy up so a re-run doesn't leave a stuck lock.
curl -sS -X POST "${ENDPOINT}/v1/predict" "${H_AUTH[@]}" "${H_JSON[@]}" \
    -d "$(jq -n --arg run_id "${new_run_id}" '{action: "cancel", run_id: $run_id}')" >/dev/null

echo
echo "🎉 All 5 curl checks passed against ${ENDPOINT}."
