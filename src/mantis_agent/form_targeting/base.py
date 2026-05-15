"""Protocol + result shape for form-target grounding (#406).

The form handler (``gym/step_handlers/form.py``) used to call three
methods on :class:`ClaudeExtractor` directly. Splitting those out
into a protocol lets us:

1. Swap implementations without touching the form handler — a Holo3-
   backed provider is added in :mod:`.holo3` so a Claude API
   overload doesn't have to stall every form step.
2. Mock the surface cleanly in tests instead of monkeypatching the
   extractor.
3. Move toward a future where ``extract_data`` / ``verify_gate`` stay
   on Claude (they need prose reasoning) while grounding can route
   to whichever VLM is cheapest / least overloaded.

Result-shape choice: a plain ``dict`` with documented keys, not a
dataclass. The form handler reads ``result["x"]`` / ``result["y"]``
/ ``result["action"]`` / ``result["label"]`` / ``result["value"]``
today, and the legacy methods returned that same dict shape. The
:class:`FormTargetResult` ``TypedDict`` documents the contract
without forcing a migration of all read sites.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

from PIL import Image


class FormTargetResult(TypedDict, total=False):
    """Shape returned by :meth:`FormTargetProvider.find_form_target` and
    :meth:`FormTargetProvider.find_target_by_affordance`.

    ``total=False`` because legacy callers only required ``x`` / ``y``
    and the provider may omit ``value`` for affordance results (no
    label match means no canonical value to type). Documenting
    expected keys without breaking callers that read with ``.get(...)``.
    """

    x: int
    y: int
    action: str   # "click" | "right_click" | "type" | "select" | "not_found"
    label: str
    value: str


class DropdownVerifyResult(TypedDict):
    """Shape returned by :meth:`FormTargetProvider.verify_dropdown_value`."""

    matches: bool
    observed: str


@runtime_checkable
class FormTargetProvider(Protocol):
    """Locate form elements (input / button / dropdown / option) on
    any page and return pixel coordinates.

    Every method returns ``None`` for the "no usable result" path so
    the form handler's existing fall-through logic (scroll-probe →
    affordance fallback → ``form_target_not_found``) keeps working
    regardless of which provider is wired in.

    Implementations:

    - :class:`ClaudeFormTargetProvider` — sends a screenshot to Claude
      via the shared :class:`AnthropicToolUseClient` and parses a
      tool_use response.
    - :class:`Holo3FormTargetProvider` — sends the screenshot + a
      grounding prompt to the existing Holo3 brain endpoint and parses
      the click coordinates Holo3 emits.

    ``runtime_checkable`` so test code can use ``isinstance(x,
    FormTargetProvider)`` for assertion clarity without forcing
    explicit subclassing.
    """

    def find_form_target(
        self,
        screenshot: Image.Image,
        intent: str,
        *,
        target_label: str = "",
        target_value: str = "",
        target_aliases: list[str] | None = None,
        region: Any = None,
    ) -> FormTargetResult | None:
        """Locate a labelled form element.

        Args:
            screenshot: Current page screenshot.
            intent: Free-text description ("Click the user ID input
                field and enter alice", "Click the Login button").
            target_label: Structured label from
                ``MicroIntent.params["label"]`` (preferred — more
                reliable than parsing free text).
            target_value: Optional value to type / option to select.
                The runner re-reads this from ``params`` for actual
                typing, but providing it here helps disambiguate.
            target_aliases: Alternate labels (e.g. ``["Update",
                "Save", "Save Changes"]``) — any alias is a valid
                visual match.

        Returns:
            dict with keys ``x`` / ``y`` / ``action`` (``click`` |
            ``type`` | ``select`` | ``right_click``) / ``value`` /
            ``label``. ``None`` on failure (caller treats as not-found).
        """
        ...

    def find_target_by_affordance(
        self,
        screenshot: Image.Image,
        intent: str,
    ) -> FormTargetResult | None:
        """Locate an element by visual affordance — shape, position,
        styling — independent of label text. Used as the fallback when
        :meth:`find_form_target` exhausts on label match.

        The ``intent`` verb drives the element-type pick (``click`` /
        ``type`` / ``select`` / ``toggle``). Returns the same result
        shape; ``None`` when no plausible element is on screen.
        """
        ...

    def verify_dropdown_value(
        self,
        screenshot: Image.Image,
        dropdown_label: str,
        expected_value: str,
    ) -> DropdownVerifyResult | None:
        """Read the current value of a (closed) dropdown and report
        whether it matches the expected option.

        Used by the ``select_option`` handler as a post-click verifier
        — after the runner picks an option, the menu closes and the
        dropdown shows the picked value. The handler screenshots that
        state and compares against the requested option to catch
        cases where the click landed on a different option (canonical
        case: y-coordinate disambiguation between adjacent menu items).

        Returns ``None`` on API failure — caller should treat that as
        "could not verify; trust the click happened" rather than
        forcing a retry on every API blip.
        """
        ...
