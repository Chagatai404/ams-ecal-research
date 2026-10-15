"""The claim template of gate SV-8: every confirmatory claim is written as estimand -> null -> statistic -> uncertainty -> assumptions
-> acceptance -> failure interpretation, before any data of the claim is read (review section 3).

``ClaimSpecification`` refuses an empty field, requires the endpoint family (P1..P4) and one NAMED primary statistic (not "one
pre-specified statistic"), and renders the table row used in registrations. It records nothing about results: the specification is
what a pre-registration commits to; ``frozen_digest`` gives a hash of the text so a later change is visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

FAMILIES = ("P1", "P2", "P3", "P4", "P5")  # P5 (efficiency) added 2026-10-12: efficiency is accepted scope, not a side result
FIELDS = ("estimand", "null", "statistic", "uncertainty", "assumptions", "acceptance", "failure_interpretation")


class IncompleteClaimError(ValueError):
    """A claim lacks a required field, so it cannot be registered."""


@dataclass(frozen=True)
class ClaimSpecification:
    claim: str
    family: str  # P1 physics difference, P2 readout survival, P3 simulator fidelity, P4 realistic performance, P5 efficiency
    primary_statistic: str  # the NAMED primary: statistic, window, representation, energies
    estimand: str
    null: str
    statistic: str
    uncertainty: str
    assumptions: str
    acceptance: str
    failure_interpretation: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise IncompleteClaimError(f"family must be one of {FAMILIES}, got {self.family!r}")
        missing = [name for name in ("claim", "primary_statistic", *FIELDS) if not str(getattr(self, name)).strip()]
        if missing:
            raise IncompleteClaimError(f"claim {self.claim!r} is missing: {', '.join(missing)}")
        vague = ("pre-specified statistic", "to be decided", "tbd")
        if any(v in self.primary_statistic.lower() for v in vague):
            raise IncompleteClaimError("the primary must be NAMED (statistic, window, representation, energies), not described")

    def markdown_row(self) -> str:
        cells = [self.claim, self.family, self.primary_statistic, *(getattr(self, f) for f in FIELDS)]
        return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"

    def frozen_digest(self) -> str:
        """SHA-256 of the canonical text; store it in the registration so any later edit of the claim is detectable."""

        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def markdown_table(claims: list[ClaimSpecification]) -> str:
    header = "| claim | family | named primary | " + " | ".join(f.replace("_", " ") for f in FIELDS) + " |"
    rule = "|" + "---|" * (3 + len(FIELDS))
    return "\n".join([header, rule, *(c.markdown_row() for c in claims)])
