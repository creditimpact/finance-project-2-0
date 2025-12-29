# Phrases Bank

Reusable phrases for letters are stored in YAML.
This guide covers the schema, adding new variants, and masking rules.

## YAML Schema

Each entry looks like this:

```yaml
- id: dispute-account
  text: "I dispute the accuracy of {account}"
  variants:
    - "I contest the report for {account}"
    - "The information for {account} is false"
  mask:
    - account
```

- `id` – unique identifier for the phrase.
- `text` – base phrase with placeholders.
- `variants` – optional list of alternate phrasings.
- `mask` – placeholders that must be masked before model prompts or logging.

## Adding Variants Safely

1. Copy the structure above.
2. Preserve placeholders; each variant must contain the same `{}` tokens as `text`.
3. Run unit tests and link checks before committing.
4. Avoid slang or unreviewed legal language.

## Masking Rules

Values listed in `mask` are replaced with `***` when serializing.
Mask every field that could contain PII or customer-specific data.
