# Integrating any agent with Mantis

This is the **agent-side integration playbook** — a prescriptive checklist
for plugging *any* third-party agent (OpenAI Computer Use, Anthropic
Computer Use, Voyager, AutoGPT-style frameworks, your own bespoke loop)
into a Mantis backend without rediscovering the integration mistakes
that produce silent halts.

> Companion reading:
>
> - [Generic CUA over HTTP](generic-cua.md) — the simplest shape (no
>   library, no env wiring; just `/v1/predict`).
> - [Embedding MicroPlanRunner](embedding-microplanrunner.md) — the
>   library shape, where you drive the runner in your own process.

---

## Pick your shape

Two integration shapes; pick by what your agent already owns.

| Shape | When | What you write |
|---|---|---|
| **HTTP client only** | Your agent has no browser stack and just needs extraction. | A 30-line client that POSTs to `/v1/predict`, polls until done, reads the result. See [Generic CUA](generic-cua.md). |
| **Library embed with your own env** | Your agent owns the desktop / Xvfb / Chrome stack and wants Mantis as the brain. | A `GymEnvironment` adapter (~150 LoC) + a `MicroPlanRunner` driver. See [Embedding MicroPlanRunner](embedding-microplanrunner.md). |

If you're unsure, start with HTTP. It's harder to hit silent integration
bugs because the runtime contract is enforced server-side.

---

## The runtime contract — five things your env must do

These are the contracts the runner relies on. Most integration halts trace
back to a violation of one of these. They apply to the **library embed**
shape; the HTTP shape inherits them from Mantis's own xdotool env.

### 1. `current_url` is a `@property`, not a method

The runner reads `self.env.current_url` as a property. If you define it as
a method:

```python
# WRONG — bound method is truthy, runner returns it as if it were the URL.
def current_url(self) -> str:
    return self._cdp_get_url()

# RIGHT — @property so attribute access returns the value.
@property
def current_url(self) -> str:
    return self._cdp_get_url()
```

Since [PR #91](https://github.com/mercurialsolo/mantis/pull/91) the runner
defensively detects `callable(raw)` and invokes it, but this is a fallback
— other readers (operators, dashboards, custom plan verifiers) may not.

### 2. CDP must be reachable from the runner's process

The runner queries Chrome DevTools Protocol's `/json/list` endpoint to
read the active tab's URL. Two things must be true:

- Chrome launched with `--remote-debugging-port=<port> --remote-debugging-address=127.0.0.1`.
- The runner's Python process can reach `127.0.0.1:<port>` (same network
  namespace, no firewall blocking the loopback port).

If your env is a thin wrapper around someone else's Chrome instance,
verify CDP independently before you ship:

```python
import urllib.request, json
tabs = json.loads(urllib.request.urlopen(
    "http://127.0.0.1:9222/json/list", timeout=2,
).read())
print([t["url"] for t in tabs if t["type"] == "page"])
```

Should print one or more page URLs. If empty / connection-refused, fix
that *before* writing any agent code — every URL-dependent verify path
silently falls back to OCR which is slower and less reliable.

### 3. `reset(start_url=...)` must actually navigate

The runner's `_execute_navigate` calls `env.reset(task="navigate", start_url=url)`
and expects the browser to be on `url` when it returns. Common
integration miss: a wrapper env's `reset` ignores `start_url`, expecting
the agent to issue an explicit `navigate` action afterwards.

```python
def reset(self, task: str, *, start_url: str = "", **_) -> GymObservation:
    if start_url:
        self._navigate(start_url)   # don't skip this
    return self._capture()
```

Verifiable: a unit test that calls `env.reset(start_url="https://example.com")`
and asserts `env.current_url` resolves to `https://example.com` within the
first-paint wait.

### 4. `step(action)` synthesises real input events

Particularly clicks. SPAs that bind handlers to row containers via
`addEventListener("click", …)` need a real `pointerdown` → `pointerup`
pair. Some Xvfb / xdotool integrations send only the `mousedown` half
or skip the move-and-click choreography. Symptom: every click in the
trace logs but no navigation happens.

If your env can't send real events end-to-end, escalate via CDP
`Input.dispatchMouseEvent` for the small set of actions that matter
([issue #89 §1 follow-up](https://github.com/mercurialsolo/mantis/issues/89)
tracks this). Until then, expect the click-verify cascade
(plain → middle-click → 4 probe points) to bottom out and the runner to
mark the step `CLICK FAILED`.

### 5. `screenshot()` returns a fresh image, not a cached one

The verify loop calls `env.screenshot()` after every action and assumes
the returned PIL image reflects the current display. Wrappers that cache
the last screenshot for performance — or that take the screenshot
asynchronously and return a stale frame — break extraction silently
because the OCR-based URL fallback then reads a pre-navigation address
bar.

---

## The micro-plan contract — what your agent emits

Your agent doesn't author `MicroIntent` JSON by hand. Send `plan_text`
(English) and let the server's `PlanDecomposer` turn it into a typed
plan:

```json
{
  "plan_text": "Open https://crm.example.com. Log in as alice / <password>. Open the Leads tab. Click the first qualified lead. In the edit form, set Industry Vertical to Space Exploration. Save changes.",
  "max_cost": 3.0,
  "max_time_minutes": 15,
  "detached": true
}
```

The decomposer turns that into `navigate` / `fill_field` × 2 / `submit`
× 3 / `select_option` / `submit`. Cached by hash for repeat calls.

If you're hand-authoring micro-plans (volume use cases, regression
testing), the [glossary](../reference/glossary.md) lists every step
type. Two patterns specifically to know about:

- **`navigate` first-paint wait**: slow proxied SPAs need
  `params={"wait_after_load_seconds": 35}` on the navigate step, or set
  `MANTIS_NAV_WAIT_SECONDS=35` on the deployment. The default is 18 s
  (covers Cloudflare auto-solve).
- **`submit` aliases**: primary-submit buttons whose copy varies across
  products (`Update Lead` / `Save Changes` / `Save`) should set
  `params={"label": "Update Lead", "aliases": ["Save", "Save Changes", "Update"]}`.
  The grounder treats any alias as a valid match, and the form-finder
  scrolls Page_Down up to 4× to find buttons below the fold.

Both of these are `params`-only — no env-side or library-side wiring
needed.

---

## Diagnosing failures — the log lines that matter

When something halts, the runner's trace is the source of truth. The
high-signal log lines:

| Line | Meaning |
|---|---|
| `[url] cdp=https://...` | CDP returned a URL — the verify path is healthy. |
| `[url] cdp unavailable (URLError); falling back to OCR` | Chrome isn't reachable on the CDP port. Fix the launch flags or the network namespace. |
| `[url] cdp empty; falling back to OCR` | CDP reachable but returned no tabs. Chrome started but no page is loaded yet — usually a navigate-too-fast or first-paint timing bug. |
| `[url] ocr=<empty>` | Both CDP *and* OCR returned empty. Either the page is mid-render or the address bar is occluded — most actionable as a `wait_after_load_seconds` bump. |
| `[claude-click] Plain click did not navigate` | The click landed but no navigation followed. Either the SPA needs synthetic events your env doesn't dispatch (see contract §4), or the URL-read source is wrong (see the `[url]` lines above). |
| `[claude-form] submit '<label>' not in viewport — scrolling Page_Down` | Form-finder is scrolling for a below-fold button. Healthy. |
| `[claude-form] submit: button '<label>' not found after 4 scroll(s)` | Genuine miss. Try setting `params["aliases"]` to cover copy variation. |

If you see `(url=)` empty across every retry of a click step, the
`[url] …` lines tell you exactly which leg to fix. Without them, you're
guessing.

---

## The five integration mistakes (and how to detect them)

These are the issues real-world integrations have surfaced — preflight your
integration against them.

| Mistake | Detection | Fix |
|---|---|---|
| `current_url` defined as method instead of `@property` | Trace shows `(url=)` empty even when CDP is reachable. The bound-method object short-circuits the truthy check. | Change to `@property`. PR #91 also detects callable and invokes it. |
| Chrome launched without `--remote-debugging-port` | `[url] cdp unavailable (URLError)` on every step. | Add the flag (and `--remote-debugging-address=127.0.0.1`) to the launch cmd. |
| `env.reset(start_url=...)` ignored | The first navigate step doesn't actually load the URL. Subsequent steps run against `about:blank` or a stale page. | Make `reset` honour `start_url`. Pin with a unit test. |
| Wrapper `screenshot()` returns cached frame | OCR reads the pre-navigation address bar. CDP reads the new URL. The two disagree, often silently. | Force a fresh capture per call. |
| Pip-installed wheel doesn't match the SHA tag | Code at `+abc1234` doesn't match `git show abc1234`. Build pipeline cached an older sdist. | Rebuild without cache. Verify with `pip show mantis-agent` + a single line `grep` on the installed file. |

---

## Pre-flight checklist

Before claiming your integration works, run all of these:

- [ ] `env.current_url` is a `@property` (not a method).
- [ ] CDP `curl http://127.0.0.1:9222/json/list` returns the current
      page from inside the runner's container / process.
- [ ] `env.reset(start_url="https://example.com")` lands on
      `https://example.com` (assert via `env.current_url`).
- [ ] `env.step(Action(CLICK, ...))` synthesises a real click — verify
      with a test page that `console.log`s on `pointerdown` and
      `pointerup`.
- [ ] `env.screenshot()` returns a fresh image — diff two consecutive
      calls after a `KEY_PRESS Page_Down` and confirm they differ.
- [ ] Your `mantis-agent` install matches the SHA you intended:
      `python -c "from mantis_agent.gym.micro_runner import MicroPlanRunner; import inspect; print(inspect.getsourcefile(MicroPlanRunner))"`
      then `grep _read_current_url <that-file>` should hit.
- [ ] A trivial `plan_text` round-trip succeeds end-to-end (HTTP shape) or
      a `MicroPlan` round-trip succeeds end-to-end (library shape).
- [ ] The trace shows `[url] cdp=...` on at least one step.

If every box is ticked, you've avoided the known-failure surface. New
failure modes go to the [issue tracker](https://github.com/mercurialsolo/mantis/issues) —
favour reproducers with the offending `[url]` / `[claude-click]` /
`[claude-form]` log lines attached.

---

## Versioning and SHA pinning

Pin `mantis-agent` to a specific tag or git SHA in your agent's lockfile.
The decomposer prompt is versioned (`v13_submit_aliases` at time of
writing); cached plans regenerate automatically when the version bumps,
so you don't need cache-busting code on your side, but you do want
your agent's behaviour reproducible across deploys.

If you're tracking a Mantis branch (e.g. for an integration that hasn't
landed in `main` yet), use `MANTIS_GIT_SHA=<sha>` as the build arg the
deployer reads — that pattern keeps `pip` cache-aware while letting the
SHA tag remain accurate.

---

## Plugging in your own brain

Mantis ships several reference brains (`holo3`, `claude`, `opencua`,
`llamacpp`, `gemma4`, `agent-s`) registered under the
`mantis_agent.brain_protocol` registry. Swap one of these — or add your
own — without forking the runtime.

### 1. Pick a built-in by name

Set `MANTIS_BRAIN` on the deployment:

```bash
export MANTIS_BRAIN=claude     # surgical-reasoning only
# or
export MANTIS_BRAIN=opencua    # local OpenCUA-7B endpoint
```

The legacy `MANTIS_MODEL` env var keeps working for one minor release.
When both are set, `MANTIS_BRAIN` wins. The legacy alias `gemma4-cua`
maps to `gemma4`.

### 2. Register your own backend

Implement the `Brain` protocol — `load()` plus `think(frames, task,
action_history=None, screen_size=(1920, 1080))` returning an
`InferenceResult`-shaped object — then register it before the runtime
starts:

```python
from mantis_agent import Brain, register_brain


class MyBrain:
    def load(self) -> None:
        ...

    def think(self, frames, task, action_history=None, screen_size=(1920, 1080)):
        ...


register_brain("my-brain", lambda: MyBrain())
```

Then run with `MANTIS_BRAIN=my-brain`. The runtime asks
`mantis_agent.resolve_brain("my-brain")` and gets a fresh instance.

The protocol is `runtime_checkable`, so `isinstance(b, Brain)` works for
quick smoke tests without forcing inheritance from a base class.

`register_brain` and `resolve_brain` are pure typing + dict — they pull
no GPU / API dependencies, so a plugin package can register a brain on
import without growing the slim install.

---

## Where to go next

- [Recipes](recipes.md) — copy-paste micro-plans for jobs, e-commerce,
  news, real-estate.
- [Embedding MicroPlanRunner](embedding-microplanrunner.md) — full
  Python surface for the library shape.
- [Reference glossary](../reference/glossary.md) — every step type,
  every helper, every env var.
- [Issue tracker](https://github.com/mercurialsolo/mantis/issues) —
  filing a bug? Include the `[url] …` log lines.
