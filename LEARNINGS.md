# Mantis CUA Agent — Learnings & Experiment Log

_Updated: 2026-04-17 | 180+ commits | ~$165+ GPU spend | Branch: feat/gym-anything-integration_

---

## Executive Summary

We're building a CUA (Computer Use Agent) that can extract leads from boat listing sites using open-weight models instead of proprietary APIs. We've achieved 83.3% on OSWorld benchmarks (OS domain) and can successfully extract leads from BoatTrader, but the economics don't work yet — most private sellers don't post phone numbers (~5-10% hit rate), and each lead costs $4-29 depending on model/run.

---

## What's Working

### 1. XdotoolGymEnv — Zero-Fingerprint Browser Automation
- Real Chrome + Xvfb + xdotool = undetectable by Cloudflare/bot detection
- X11 events are indistinguishable from human input
- Screenshots via mss/scrot (pixel capture, no browser API)
- **Proven**: No Cloudflare blocks once we switched from Playwright/CDP
- **Key lesson**: Every automation framework (Playwright, CDP, Puppeteer) leaks signals. Only OS-level input injection is truly undetectable.

### 2. Gemma4 on OSWorld — 83.3% (20/24 OS tasks)
- Gemma4 26B-A4B via llama.cpp on single A100-80GB
- pyautogui as primary action language (model was trained on it)
- Distillation loop: analyze failure → diagnose → store learning → retry
- ReliableController: transparently upgrades pyautogui.write() to xdotool for special chars
- **Key lesson**: Respect the model's training distribution. Making subprocess.run() the default dropped score from 83% to 67%.

### 3. Custom Gemma4-CUA Fine-Tune — 100% CRM in 3 Steps
- QLoRA fine-tune on 5000 AgentNet tasks, 3 epochs, rank=32
- Loss: 3.8 → 0.27
- Fastest of all models tested (3 steps vs 12 for EvoCUA-32B)
- Parse actions from `reasoning_content` (model thinks then acts)
- **Key lesson**: Small, targeted fine-tunes massively outperform prompting alone.

### 4. Reasoning Budget is Task-Dependent
- `budget=0` for CLI/terminal tasks (OSWorld) — model over-thinks on shell commands
- `budget=512` sweet spot for browser CUA — enough to reason about visual layouts
- `budget=4096` for complex form-filling — but 3-5x slower
- **Key lesson**: There's no universal budget. Match reasoning depth to task complexity.

### 5. Parallel Workers (1 Page Per Worker)
- 5x wall-clock speedup over sequential
- Each worker owns a full page (~25 listings)
- Retry 3x on crash/Cloudflare/preemption
- **Key lesson**: Don't pre-slice listings across workers. Dynamic page queue is simpler and more fault-tolerant.

### 6. Persistent Chrome Profile on Modal Volume
- Cookies survive across runs (login once, extract many times)
- Clean session files only, not full profile (avoids corruption)
- IPRoyal Miami residential proxy with sticky sessions per run

### 7. Strict Extraction Validation
- Real phone validation: 555-exchange filtered, URL fragments filtered
- Dedup by phone digits + listing URL
- Eliminates ~90% of false positives (social media links, ad tracking numbers)

### 8. Screenshot Replay — 60x Faster Prompt Iteration
- Every run now saves screenshots to Modal volume (`/data/screenshots/`)
- `ReplayGymEnv` replays cached screenshots locally without browser/GPU
- `test_extraction_prompt.py` tests prompts against screenshots via Claude API in 30sec
- `replay_test.py` CLI: download, test single prompts, run full replay
- **Key finding from screenshots**: Phone was visible on step 20 but model clicked into photo gallery and got trapped for 60 steps

### 9. RegionGrounding — Heuristic Click Safety
- Clamps click coordinates to safe content area (y>80, y<660, x>20, x<1260)
- Prevents footer social icon clicks (y>660) and header menu clicks (y<80)
- Zero overhead — no model, no CDP, just coordinate math
- Graceful fallback: if grounding fails, uses brain's original coordinates

### 10. JSON Nested Action Parsing
- Gemma4-CUA sometimes outputs `{"action":"key_press","parameters":{"keys":"alt+left"}}`
- Parser now flattens nested `parameters`/`arguments`/`params` dicts
- Also handles markdown-fenced JSON (`\`\`\`json\n{...}\n\`\`\``)
- Eliminated parse failures that were wasting 30%+ of steps

---

## What's NOT Working

### 1. Prompt Engineering Has Hit a Ceiling
- Models click Facebook/Instagram icons despite explicit "NEVER CLICK" instructions in 25-50% of iterations
- Adding more negative examples, caps, emphasis — no measurable improvement
- **Root cause**: Text instructions can't override visual saliency. Colorful social media icons are visually dominant. The model sees them and clicks them regardless of prompt.
- **Spent**: 15+ prompt iterations, multiple commit cycles, no improvement beyond ~75% adherence

### 2. The Fundamental Lead Economics Problem
- Only ~5-10% of private sellers on BoatTrader post phone numbers
- Best case: $4.26/lead (EvoCUA-32B, good proxy day)
- Worst case: $29+/lead (Gemma4 parallel, prompt-only off-site avoidance)
- 80 listings scanned → 3 unique phone leads total
- **Implication**: Even with perfect extraction accuracy, the hit rate caps economic viability. Need higher-yield sources or broader lead definition.

### 3. EvoCUA-8B Parse Failures
- 225/500 steps unparseable in brain_opencua
- The model outputs valid reasoning but action format doesn't match our parser
- EvoCUA-32B works fine with same parser — the 8B model is genuinely worse at structured output
- **Cost**: Cheapest model is unusable, forcing us to 2x A100 for 32B

### 4. E4B Planner Produces Generic Plans
- Gemma4-E4B (4B parameter model) as text planner generates instructions without visual anchors
- "Click the boat listing" vs "Click the LARGE rectangle with boat photo and price text"
- CUA models need spatially grounded instructions to avoid clicking wrong elements
- **Spent**: Multiple iterations on planner prompt, fundamental model capability issue

### 5. CDP-Based Runtime Guardrails Don't Work in xdotool Env
- JS injection for cookie dismissal, URL detection, CSS hiding
- Timing races with xdotool (event arrives before JS executes)
- WebSocket failures when Chrome is under load
- Violates the zero-fingerprint architecture we chose for good reason
- **Decision**: Removed all CDP guardrails. Model handles everything visually.

### 6. Off-Site Navigation Has No Runtime Safety Net
- CDP backtrack was removed (for good reason — fingerprinting)
- RegionGrounding clamps footer/header clicks but doesn't prevent inline social links
- Off-site avoidance is prompt-based + region clamping, but ~25% still leak through
- **Mitigation**: RegionGrounding reduced from 50% to ~25%. Full fix needs Opus visual planner.

### 7. Stale URLs Cause Silent 0% Runs
- BoatTrader changed URL schema — `condition-used/type-power/price-35000,/zip-33101/radius-100/` returns 404
- `seller-private/` filter URL also returns 404
- Only `boattrader.com/boats/` (base URL) works, but shows ALL boats including dealers
- **This caused ALL recent 0% runs** — model was on 404 pages, not extraction failures
- **Fix needed**: Opus visual planner discovers correct URL by browsing, or model applies filters visually via s1_search

### 8. Image Gallery Trap
- Model clicks boat photos → enters lightbox/image viewer → spends 40+ steps trying to close
- Screenshot evidence: step 20 had phone visible, steps 40-80 stuck in gallery
- Prompt says "IGNORE the photos, scroll DOWN" but model clicks them anyway
- **Fix needed**: Grounded click model that targets listing cards specifically, not photos

---

## Why Testing & Iteration Cycles Are Slow

This is the single biggest drag on progress. Each experiment cycle takes **30-90 minutes wall clock** and costs **$4-12 in GPU**.

### The Feedback Loop

```
Code change → Modal deploy (~2min) → Container build (~3min) → Model load (~2min) →
Chrome launch → Navigate to site → Process listings (~10-20min each) →
Check results on Modal volume → Diagnose failure from logs → Repeat
```

**Total: 30-90 minutes per iteration, $4-12 per run**

### Why It's Inherently Slow

1. **No local testing possible**: The full pipeline requires A100 GPU (llama.cpp + Gemma4 26B), Xvfb display, real Chrome, and IPRoyal proxy. Can't run locally on Mac.

2. **Modal cold starts**: Every code change requires a new container build. Image building (apt-get, pip install, model download) adds 2-5 minutes before any code runs.

3. **Real website latency**: BoatTrader pages take 3-10 seconds to load through residential proxy. Can't mock this without losing the zero-fingerprint guarantee.

4. **Model inference is slow**: Gemma4 with budget=512 takes 5-15 seconds per step. 40 steps per listing × 25 listings per page = 15-20 minutes per page minimum.

5. **Failures are only visible at the end**: A prompt engineering change might look fine for the first 5 listings then fall apart on listing #6 when the page layout shifts. Need full-page runs to validate.

6. **No unit tests for visual grounding**: We can't unit-test "does the model click the right thing on a boat listing page" without actually running the model on the page. There's no synthetic benchmark for this.

7. **Log inspection is manual**: Results land on Modal volumes. `check_boattrader.sh` helps but debugging requires reading through JSON traces step-by-step.

### What We've Done to Speed Things Up

- `human_speed=False` + removed inter-iteration sleep → 2-3x faster per listing
- Parallel workers → 5x wall-clock reduction for multi-page runs
- `check_boattrader.sh` / `monitor_boattrader.sh` → live progress without waiting for completion
- `--detach` runs → start and check later instead of blocking terminal
- Fast-fail on 404/error pages → skip stale listings immediately

### What Would Actually Help (but haven't built yet)

- **Local Gemma4 inference** via llama.cpp on Mac (M-series) for prompt iteration without Modal
- **Cached screenshot replay**: Record screenshots from a real run, replay them locally to test prompt/parsing changes without GPU or network
- **Synthetic visual benchmark**: 20 annotated BoatTrader screenshots with ground-truth click targets, testable offline

---

## Hypotheses Remaining to Test

### High Priority

#### H1: Opus Visual Planner (Issue #40)
**Hypothesis**: Claude Opus browses the site first via xdotool, discovers layout visually, then generates a rich task suite with visual anchors, error handlers, and negative examples. Cheap models execute the loop.

**Why we believe this**: Prompt engineering has plateaued because models need visual ground truth from the actual site, not text descriptions. Opus has strong visual reasoning and can generate CUA-aware instructions that include spatial descriptions ("the LARGE rectangle", "small colored squares in footer = social links, NEVER click").

**Cost to test**: ~$2-5 (Opus API for planning) + existing GPU for execution
**Expected impact**: Reduce off-site navigation from 25-50% to <5%

#### H2: Cached Screenshot Replay for Fast Iteration
**Hypothesis**: Record screenshots + action traces from one real run. Replay the screenshots locally to test prompt/parsing changes without GPU, network, or real websites.

**Why we believe this**: 80% of our iteration time is waiting for infrastructure, not thinking about the problem. If prompt changes could be tested in 30 seconds instead of 30 minutes, we'd move 60x faster on the prompt engineering axis.

**Cost to test**: ~1 day of engineering
**Expected impact**: Prompt iteration cycle from 30min → 30sec

#### H3: Broader Lead Definition (Email + Contact Form)
**Hypothesis**: If we expand "viable lead" beyond phone-only to include email addresses and contact form submissions, hit rate jumps from 5-10% to 30-50%.

**Why we believe this**: Most BoatTrader sellers have a "Contact Seller" button even when they don't show phone numbers. The CUA can fill out contact forms directly.

**Cost to test**: Modify extraction validation + add form-filling task to workflow
**Expected impact**: 5-10x more leads per run at similar GPU cost

### Medium Priority

#### H4: EvoCUA-32B with Opus-Generated Plans
**Hypothesis**: EvoCUA-32B's 23% hit rate was limited by generic prompts, not model capability. With Opus-generated visual plans, it could match Gemma4-CUA's per-listing accuracy at lower cost (no fine-tune needed).

**Cost to test**: Depends on H1 (Opus planner)
**Expected impact**: Best cost/lead model if parsing stays reliable

#### H5: Gemma4-31B Dense for Browser CUA
**Hypothesis**: The 31B dense model (vs 26B MoE) has native tool calling and stronger visual grounding. Could eliminate parse failures and improve click accuracy.

**Cost to test**: ~$8-10 per test run (same A100, slightly slower inference)
**Expected impact**: Fewer parse failures, potentially better visual grounding

#### H6: Distillation from Claude Trajectories
**Hypothesis**: Run Claude (API) on BoatTrader to generate perfect trajectories. Fine-tune Gemma4 on these trajectories for site-specific CUA behavior.

**Why we believe this**: Our Gemma4-CUA fine-tune (AgentNet data) achieved 100% on CRM in 3 steps. Site-specific distillation could achieve similar gains on BoatTrader.

**Cost to test**: ~$20 Claude API + ~$15 fine-tuning on Modal
**Expected impact**: Near-Claude accuracy at open-weight cost

### Lower Priority / Exploratory

#### H7: Multi-Site Expansion
**Hypothesis**: The architecture (xdotool env + parallel workers + workflow runner) is general enough to work on other boat listing sites (YachtWorld, Boats.com) with only task file changes.

**Risk**: Each site has different layouts, anti-bot measures, and data formats.

#### H8: SoM (Set-of-Marks) for Visual Grounding
**Hypothesis**: Overlaying numbered bounding boxes on screenshots helps models identify clickable elements more accurately, reducing off-site clicks.

**Why we haven't tested**: Adds DOM dependency (need element detection), which conflicts with zero-fingerprint xdotool approach. Would need a vision-only SoM (e.g., YOLO-based element detection on screenshots).

#### H9: Reward Model for Self-Evaluation
**Hypothesis**: Train a small model to evaluate "did this step make progress?" from before/after screenshots. Use as a runtime guardrail to detect and recover from off-site navigation.

**Cost to test**: Significant — need labeled data + training pipeline
**Expected impact**: Runtime safety net without CDP/DOM dependency

---

## Decision Log

| Date | Decision | Outcome |
|------|----------|---------|
| Apr 8 | Start with Playwright for browser CUA | FAILED — Cloudflare blocks |
| Apr 9 | Switch to CDP (connect to real Chrome) | PARTIAL — works but leaks signals |
| Apr 10 | Switch to xdotool + Xvfb | SUCCESS — undetectable |
| Apr 10 | Use EvoCUA-8B for cheap extraction | FAILED — 225/500 parse failures |
| Apr 11 | Fine-tune Gemma4 on AgentNet | SUCCESS — 100% CRM in 3 steps |
| Apr 12 | Use Gemma4-E4B as text planner | FAILED — generic instructions |
| Apr 13 | Add CDP guardrails back into xdotool env | FAILED — timing races, fingerprint risk |
| Apr 14 | Pure prompt-based off-site avoidance | PARTIAL — 50-75% effective |
| Apr 15 | Parallel workers (1 page each) | SUCCESS — 5x speedup |
| Apr 15 | EvoCUA-32B for volume extraction | SUCCESS — $4.26/lead on good runs |
| Apr 16 | Gemma4-CUA budget=512 for accuracy | SUCCESS — best per-listing accuracy |
| Apr 17 | Fix 10 code bugs + fast-fail 404s | SUCCESS — reduced waste significantly |
| Apr 17 | Prompt engineering for off-site avoidance | FAILED — ceiling reached |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total commits | 170 |
| Total GPU spend | ~$130+ |
| OSWorld best (OS domain) | 83.3% (20/24) |
| CRM best (Gemma4-CUA) | 100% in 3 steps |
| BoatTrader listings scanned | ~80 |
| Unique phone leads found | 3 |
| Best cost/lead | $4.26 (EvoCUA-32B) |
| Worst cost/lead | $29+ (Gemma4 parallel) |
| Phone number hit rate | ~5-10% |
| Off-site click waste | 25-50% of iterations |
| Iteration cycle time | 30-90 minutes |
| Iteration cost | $4-12 per run |
