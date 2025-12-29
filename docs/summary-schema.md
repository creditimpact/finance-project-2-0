# Summary Schema Notes

## Validation Requirement Arguments

- Each entry in `validation_requirements.findings[]` MAY include an `argument.seed` object when the finding resolves to a strong decision (for example, `strong_actionable` or `strong`).
- Seed arguments provide concise, field-specific rationales that are stored alongside the finding for downstream use.

## Arguments Aggregation

- The account summary root MUST contain an `arguments` object.
  - `arguments.seeds[]` is a flattened, de-duplicated collection of every strong seed argument emitted from validation findings across the account.
  - `arguments.composites[]` is reserved for Strategy-layer outputs and should remain untouched by validation.

