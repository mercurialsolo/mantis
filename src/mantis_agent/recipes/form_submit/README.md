# Recipe: form_submit

Fills and submits a public web form, then captures the echoed response
page so the caller can verify the submission landed. Useful as a smoke
test for form-driven flows (login screens, contact forms, settings
panels) without writing site-specific selectors.

## What it does

1. Open the form URL.
2. Fill each field via Holo3 (`Customer name`, `Telephone`, `Email
   address`, and a `Medium` radio choice in the shipped plan).
3. Click the submit button.
4. Capture the response page contents as a single string into
   `submitted_payload`.

The shipped plan targets `httpbin.org/forms/post`, which echoes the
submitted form back — useful for asserting that field-fill + submit
worked without depending on a stateful backend.

## Tested against

- `httpbin.org/forms/post` (the canonical public form-echo endpoint).
- Adapts to similar single-page forms with distinct visible labels. To
  retarget, edit the `navigate.url` and each `holo3.hint` in
  `plan.json` to match your form's field labels.

## Known limits

- Field labels in the plan's `hint` strings (`"Click the 'Customer
  name' input, then type 'Mantis Test'"`) are how Holo3 locates the
  input. Forms with non-unique or non-visible labels (e.g. inputs
  identified only by `placeholder`) may need ClaudeGrounding turned on
  (`"grounding": true`) per step.
- Multi-step / wizard-style forms (CAPTCHA + reCAPTCHA + multi-page
  submission) aren't covered — this recipe assumes a single-page form.
- The capture step records the raw page text. If your form redirects
  to a JSON-only success endpoint, swap the final step's
  `schema_name` / `fields` to extract structured fields instead.

## Usage

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @src/mantis_agent/recipes/form_submit/plan.json
```
