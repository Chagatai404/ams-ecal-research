"""The pure helpers of the EM initialisation dump (no Geant4 needed): what is called, what is skipped, what differs."""

from typing import ClassVar

from ams_ecal.geant4_simulation import em_variant_initialisation_dump as dump


class FakeParameters:
    called: ClassVar[list[str]] = []

    def ApplyCuts(self):
        return True

    def MscRangeFactor(self):
        return 0.2

    def LateralDisplacement(self):
        return True

    def LivermoreDataDir(self):
        return "epics_2017"

    def SetApplyCuts(self, flag):
        FakeParameters.called.append("SetApplyCuts")

    def DumpAndExit(self):
        FakeParameters.called.append("DumpAndExit")

    def NeedsAnArgument(self, x):
        return x

    def ReturnsAnObject(self):
        return object()


def test_scalar_getters_call_only_getters_and_list_what_they_skipped():
    values = dump.scalar_getters(FakeParameters())
    assert values["ApplyCuts"] is True and values["MscRangeFactor"] == 0.2 and values["LivermoreDataDir"] == "epics_2017"
    assert FakeParameters.called == []
    assert set(values["_skipped"]) == {"NeedsAnArgument", "ReturnsAnObject"}
    assert not any(k.startswith(("Set", "Dump")) for k in values)


def test_differences_report_changed_parameters_and_process_names_only():
    reference = {
        "em_parameters": {"ApplyCuts": False, "MscRangeFactor": 0.04, "LPM": True, "_skipped": ["a"]},
        "processes": {"e-": ["eBrem", "eIoni", "msc"]},
    }
    other = {
        "em_parameters": {"ApplyCuts": True, "MscRangeFactor": 0.2, "LPM": True, "_skipped": ["b"]},
        "processes": {"e-": ["eBrem", "eIoni", "msc", "ePairProd"]},
    }
    result = dump.differences(reference, other)
    assert result["em_parameters"] == {"ApplyCuts": [False, True], "MscRangeFactor": [0.04, 0.2]}
    assert result["processes"]["e-"] == {"only_in_reference": [], "only_in_other": ["ePairProd"]}


def test_printed_table_comparison_sees_what_the_scalar_getters_cannot():
    base = "Step function for e+-   (0.2, 1 mm)\nType of msc step limit algorithm for e+-  1\nApply cuts on all EM processes   0\nMax 2J for sampling   10\n"
    other = "Step function for e+-   (0.8, 1 mm)\nType of msc step limit algorithm for e+-  0\nApply cuts on all EM processes   0\nMax 2J for sampling   99\n"
    result = dump.printed_table_differences(base, other)
    assert result["only_in_reference"] == [
        "Step function for e+-   (0.2, 1 mm)",
        "Type of msc step limit algorithm for e+-  1",
    ]
    assert len(result["only_in_other"]) == 2 and not any("Max 2J" in line for line in result["only_in_other"])  # noise filtered
    assert dump.printed_table_differences(base, base) == {"only_in_reference": [], "only_in_other": []}


def test_identical_dumps_have_no_differences():
    same = {"em_parameters": {"LPM": True}, "processes": {"e-": ["msc"]}}
    result = dump.differences(same, same)
    assert result["em_parameters"] == {} and result["processes"]["e-"] == {"only_in_reference": [], "only_in_other": []}
