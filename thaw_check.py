from types import MappingProxyType
from backend.core.orchestrators import _thaw

orig = {"bureaus": [{"x": 1}], "flags": []}
frozen = MappingProxyType(orig)

dup = _thaw(frozen)
dup["bureaus"][0]["x"] = 2
dup["flags"].append("touched")

assert orig["bureaus"][0]["x"] == 1, "orig mutated!"
assert orig["flags"] == [], "orig flags mutated!"
print("THAW_OK")
