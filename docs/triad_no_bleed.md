# Stage-A triad band guard

Stage-A assembles triad rows strictly from the geometric bands detected on the
corresponding header line. Each bureau’s column is defined by the header `x0`
coordinates and evaluated using closed-open intervals:

- label: `[0, tu_left_x0)`
- TransUnion: `[tu_left_x0, xp_left_x0)`
- Experian: `[xp_left_x0, eq_left_x0)`
- Equifax: `[eq_left_x0, next_label_x0)`

A small boundary guard (`TRIAD_BOUNDARY_GUARD`, default `2.0` px) is applied to
the right side of every band. Tokens that fall within that guard distance stay
with the bureau on the left, preventing seam-adjacent words from bleeding into
the next bureau when minor PDF rounding shifts occur.

Value assembly is geometry-only: only tokens classified into a bureau’s band on
that row contribute to its value. Explicit dash placeholders (`--`, em-dash,
en-dash) inside a bureau’s band become the final value for that bureau and
remain as `"--"` while still counting as empty for presence checks. A blank
band without a dash yields an empty string.

The space-delimited tail split heuristic now runs only as a last resort when a
labeled row has exactly one token after the label and no geometric hits for any
bureau. When triggered, the token is split into three segments by whitespace,
assigned to bureaus using geometry, and then the guard and dash rules above are
applied.
