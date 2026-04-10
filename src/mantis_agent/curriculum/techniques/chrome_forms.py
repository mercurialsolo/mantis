"""Chrome forms: text inputs, checkboxes, dropdowns, submit."""

NAME = "Chrome forms"
TAGS = [
    "chrome", "browser", "web", "form", "input", "field", "submit", "click",
    "text", "checkbox", "radio", "dropdown", "select", "login",
    "search", "query", "type", "enter", "button",
]
TRIGGERS = [
    r"\bform\b", r"\binput\b", r"\bfield\b", r"\bsubmit\b",
    r"\bcheckbox\b", r"\bradio button\b", r"\bdropdown\b",
    r"\blogin\b", r"sign.?in", r"\bregister\b",
    r"fill (in|out)", r"enter.*\b(name|email|password|address|phone)\b",
    r"search (for|the)", r"type.*\b(into|in)\b",
]
ALWAYS = False

CONTENT = """\
Chrome form interaction techniques:
- Click directly on the input field's center to focus it, then `type_text('the value')`. Do NOT use `key('Tab')` to reach a field unless you can see the focus state in the screenshot.
- After typing into a field, you have three submit options:
  1. `key('Return')` — works for most search forms and single-field inputs
  2. Click the explicit Submit/Search/Login button (preferred for multi-field forms)
  3. `key('Tab')` to next field if you have more to fill in
- Checkboxes and radio buttons: just `click(x, y)` on the box. The state should toggle visibly in the next screenshot.
- Dropdowns (select elements): `click(x, y)` to open it, then click the desired option from the expanded list. For native HTML selects, you can also press the option's first letter to jump.
- Clear an input before typing: focus it, then `key('ctrl+a')` to select all, `key('Delete')`, then type the new value.
- Form submission usually triggers a navigation — `wait(2)` after submit, then verify the result page in the next screenshot.
- File uploads open a native picker dialog — these are HARD to control and usually skip-worthy unless the task specifically requires them.\
"""
