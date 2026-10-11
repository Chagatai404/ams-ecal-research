"""The claim template refuses incomplete or vague claims and makes any later edit visible."""

import dataclasses

import pytest

from ams_ecal.inference import claim_specification as cs


def complete(**changes):
    base = {
        "claim": "Interacting protons are more concentrated than electrons at equal occupancy",
        "family": "P1",
        "primary_statistic": "D2 over box sides 2-3-6-9, deposition, 20 GeV, eligible in both",
        "estimand": "difference of the D2 distribution between classes among events matched on occupied cells",
        "null": "same distribution given the matching",
        "statistic": "stratum-weighted KS, five occupancy strata",
        "uncertainty": "label permutation within strata, 10,000 draws; Holm across families",
        "assumptions": "exchangeability within strata; eligibility explicit",
        "acceptance": "adjusted p <= 0.05 and the sign persists across the EM and hadronic variants",
        "failure_interpretation": "no difference beyond occupancy and morphology",
    }
    base.update(changes)
    return cs.ClaimSpecification(**base)


def test_a_complete_claim_renders_a_table_row_and_a_stable_digest():
    claim = complete()
    assert claim.markdown_row().startswith("| Interacting protons") and claim.markdown_row().count("|") >= 11
    assert claim.frozen_digest() == complete().frozen_digest()
    assert claim.frozen_digest() != complete(acceptance="adjusted p <= 0.10").frozen_digest()
    assert "named primary" in cs.markdown_table([claim]).splitlines()[0]


@pytest.mark.parametrize(
    "field", ["estimand", "null", "statistic", "uncertainty", "assumptions", "acceptance", "failure_interpretation"]
)
def test_every_field_of_the_template_is_required(field):
    with pytest.raises(cs.IncompleteClaimError, match=field):
        complete(**{field: "   "})


def test_the_family_must_be_one_of_the_four_endpoint_families():
    with pytest.raises(cs.IncompleteClaimError, match="family"):
        complete(family="P5")


def test_the_primary_must_be_named_not_described():
    with pytest.raises(cs.IncompleteClaimError, match="NAMED"):
        complete(primary_statistic="one pre-specified statistic")


def test_a_registered_claim_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        complete().acceptance = "changed after the result"  # type: ignore[misc]
