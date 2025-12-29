"""Name normalization helpers for creditors and bureaus."""

from __future__ import annotations
import logging

import re

BUREAUS = ["Experian", "Equifax", "TransUnion"]

# Allow a few common variations when looking up bureaus
BUREAU_ALIASES = {
    "transunion": "TransUnion",
    "trans union": "TransUnion",
    "tu": "TransUnion",
    "experian": "Experian",
    "exp": "Experian",
    "ex": "Experian",
    "equifax": "Equifax",
    "eq": "Equifax",
    "efx": "Equifax",
}

# Canonical creditor names and common aliases
COMMON_CREDITOR_ALIASES = {
    "citi": "citibank",
    "citicard": "citibank",
    "citi bank": "citibank",
    "cbna": "citibank",
    "bofa": "bank of america",
    "boa": "bank of america",
    "bk of amer": "bank of america",
    "bank of america": "bank of america",
    "capital one": "capital one",
    "cap one": "capital one",
    "cap1": "capital one",
    "capital 1": "capital one",
    "cap 1": "capital one",
    "chase": "chase bank",
    "jp morgan chase": "chase bank",
    "jpm chase": "chase bank",
    "wells": "wells fargo",
    "wells fargo": "wells fargo",
    "us bank": "us bank",
    "usbank": "us bank",
    "usaa": "usaa",
    "ally": "ally bank",
    "ally financial": "ally bank",
    "synchrony": "synchrony bank",
    "synchrony financial": "synchrony bank",
    "syncb": "synchrony bank",
    "paypal credit": "paypal credit (synchrony)",
    "barclay": "barclays",
    "barclays": "barclays",
    "discover": "discover",
    "comenity": "comenity bank",
    "comenity bank": "comenity bank",
    "td": "td bank",
    "td bank": "td bank",
    "pnc": "pnc bank",
    "pnc bank": "pnc bank",
    "regions": "regions bank",
    "truist": "truist",
    "bbt": "bb&t (now truist)",
    "suntrust": "suntrust (now truist)",
    "avant": "avant",
    "upgrade": "upgrade",
    "sofi": "sofi",
    "earnest": "earnest",
    "upstart": "upstart",
    "marcus": "marcus by goldman sachs",
    "goldman": "marcus by goldman sachs",
    "toyota": "toyota financial",
    "nissan": "nissan motor acceptance corp.",
    "ford": "ford credit",
    "honda": "honda financial services",
    "hyundai": "hyundai motor finance",
    "kia": "kia motors finance",
    "tesla": "tesla finance",
    "navient": "navient",
    "great lakes": "great lakes (nelnet)",
    "mohela": "mohela",
    "aes": "aes (american education services)",
    "fedloan": "fedloan servicing",
    "credit one": "credit one bank",
    "credit one bank": "credit one bank",
    "creditonebnk": "credit one bank",
    "first premier": "first premier bank",
    "mission lane": "mission lane",
    "ollo": "ollo card",
    "reflex": "reflex card",
    "indigo": "indigo card",
    "merrick": "merrick bank",
    "hsbc": "hsbc",
    "bmw financial": "bmw financial",
    "bmw fin svc": "bmw financial",
    "bmw finance": "bmw financial",
    "amex": "american express",
    "american express": "american express",
    "american express bank": "american express",
    "american express national bank": "american express",
    "santander": "santander bank",
    "santander bank": "santander bank",
    "gs bank usa": "gs",
    "bbva": "bbva usa",
    "bbva usa": "bbva usa",
    "fifth third": "fifth third bank",
    "fifth third bank": "fifth third bank",
    "onemain": "onemain financial",
    "one main": "onemain financial",
    "onemain financial": "onemain financial",
    "lending club": "lendingclub",
    "lendingclub": "lendingclub",
    "freedom financial": "freedomplus",
    "freedom plus": "freedomplus",
    "freedomplus": "freedomplus",
    "webbank fhut": "webbank fingerhut",
    "webbnk fhut": "webbank fingerhut",
    "webbank fingerhut": "webbank fingerhut",
}


logger = logging.getLogger(__name__)


def normalize_creditor_name(raw_name: str) -> str:
    """Return a cleaned, canonical creditor name."""

    name = raw_name.lower().strip()
    for alias, canonical in COMMON_CREDITOR_ALIASES.items():
        if alias in name:
            if canonical != name:
                logger.debug("Alias match: %r -> %r", raw_name, canonical)
            return canonical
    name = re.sub(r"\b(bank|usa|na|n.a\.|llc|inc|corp|co|company)\b", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def canonicalize_creditor(name: str) -> str:
    """Public helper wrapping :func:`normalize_creditor_name`."""

    return normalize_creditor_name(name)


def normalize_bureau_name(name: str | None) -> str:
    """Return canonical bureau name for various capitalizations/aliases."""
    if not name:
        return ""
    key = name.strip().lower()
    return BUREAU_ALIASES.get(key, name.title())
