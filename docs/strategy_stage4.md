# Strategy Stage 4

Stage 4 executes the final LLM strategy with policy enforcement and caching.

## Inputs
- **Prompt** – consolidated instructions passed to the LLM.
- **Stage 3 output** – structured context from earlier stages.
- **Rulebook & guardrails** – policy definitions for enforcement.

## Enforcement Steps
1. Submit the prompt to the LLM and parse the JSON reply.
2. Validate required fields and coerce types.
3. Run rulebook checks and apply deterministic fixes.
4. Record any violations and surface errors.

## Guardrails
- Normalizes entity names and numbers.
- Caps token and character length.
- Filters unsafe admissions and PII.
- Prevents forbidden dispute frames.

## Cache
- Deterministic key derived from prompt and rulebook version.
- Stores both raw LLM reply and sanitized payload.
- Hits skip re-generation; misses are inserted asynchronously.

## Outputs
- Policy-compliant strategy fragment.
- Audit log entries for each correction.
- Cache metadata indicating hit or miss.

## Metrics
- `stage4_requests_total`
- `stage4_cache_hits_total`
- `stage4_guardrail_violations_total`
- `stage4_latency_ms`
- `strategy.cache_hit`
- `strategy.cache_miss`

## Monitoring
- Dashboards should track `tokens_in`, `tokens_out`, `cost`, and `latency_ms` from
  `log_ai_request` to alert on unusual latency or cost spikes.
