"""Interactive Geant4 event display for the AMS ECAL simulation.

The installed ``geant4_pybind`` 0.1.3 wheel has no Qt (no ``G4UIQt``), so the
display uses what the wheel does ship: the Win32 UI session (``G4UIWin32``)
with an OpenGL / ToolsSG viewer, or the ``TSG_OFFSCREEN`` driver that renders
straight to a PNG without a window.

The application is the production ``ECALTransport`` built through the same
``_setup`` as ``run_batch``: production geometry, physics list, cuts and the
per-event energy / ``cos_flux`` kinematics. Nothing here edits those modules.

Launch (interactive)::

    .venv/Scripts/python.exe -m ams_ecal.geant4_simulation.event_display \
        --config configs/geant4_realistic_regime_pilot_electron.yaml --energy 50

Render one event to a PNG, no window::

    ... --offscreen out.png

Drawing another event.  Do NOT type ``/run/beamOn``: the primary generator
reads its per-event seed from a list that only ``ECALTransport.run`` fills, so
a bare ``/run/beamOn`` fails.  Use the command registered by this module::

    /ecal/drawEvent          # next event of the configured seed stream
    /ecal/drawEvent 12345    # the event with this seed (reproduces a stored event)

or re-launch with ``--seed N``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.transport_geometry import load_transport_config
from ams_ecal.geant4_simulation.geant4_backend import (
    GEOMETRY_CONFIG,
    PROJECT_ROOT,
    TRANSPORT_CONFIG,
    ECALTransport,
    EventResult,
    PilotConfigError,
    _setup,
    event_seeds,
    load_pilot_config,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geant4_realistic_regime_pilot_electron.yaml"
WARN_ABOVE_GEV = 200.0

# Distinct trajectory colours by particle (Geant4 colour names or r g b).
TRAJECTORY_COLOURS: dict[str, str] = {
    "e-": "red",
    "e+": "blue",
    "gamma": "green",
    "proton": "magenta",
    "neutron": "yellow",
    "pi+": "cyan",
    "pi-": "orange",
}
DEFAULT_TRAJECTORY_COLOUR = "grey"


def _apply(ui: Any, command: str, *, strict: bool = True) -> None:
    """Apply one UI command; ``ApplyCommand`` returns a G4UIcommandStatus (0 = success)."""

    status = ui.ApplyCommand(command)
    if strict and status not in (0, None):
        print(f"[event_display] command failed (status {status}): {command}", file=sys.stderr)


def vis_commands(
    *, driver: str, fibres: bool, offscreen: Path | None, size: int, min_track_mev: float
) -> list[str]:
    """The ordered vis command strings; kept separate so they can be inspected or tested."""

    if offscreen is not None:
        open_viewer = [
            f"/vis/open TSG_OFFSCREEN {size}x{size}",
            "/vis/tsg/offscreen/set/format zb_png",
            f"/vis/tsg/offscreen/set/file {offscreen.as_posix()}",
        ]
    elif driver == "OGL":
        open_viewer = [f"/vis/open OGLSWin32 {size}x{size}"]
    else:
        open_viewer = [f"/vis/open TSG_WINDOWS_GLES {size}x{size}"]

    commands = [
        *open_viewer,
        # Make the world and envelopes invisible; superlayers are faint translucent slabs (the lead/glue matrix) so the
        # fibres, drawn in a saturated colour, stay readable.
        "/vis/viewer/set/autoRefresh false",
        "/vis/drawVolume",
        "/vis/geometry/set/visibility World 0 false",
        "/vis/geometry/set/visibility ECAL 0 false",
        "/vis/geometry/set/visibility Extension 0 false",
        "/vis/geometry/set/colour Superlayer_x 0 0.55 0.55 0.65 0.12",
        "/vis/geometry/set/colour Superlayer_y 0 0.65 0.55 0.55 0.12",
        "/vis/geometry/set/visibility Superlayer_x 0 true",
        "/vis/geometry/set/visibility Superlayer_y 0 true",
        f"/vis/geometry/set/visibility Fibre 0 {'true' if fibres else 'false'}",
        "/vis/geometry/set/colour Fibre 0 0.95 0.75 0.2 0.35",
        "/vis/viewer/set/style surface",
        "/vis/viewer/set/hiddenMarker true",
        "/vis/viewer/set/upVector 0 1 0",
        "/vis/viewer/set/viewpointThetaPhi 55 35 deg",
        "/vis/viewer/zoomTo 1.4",
        "/vis/scene/add/trajectories smooth",
        "/vis/modeling/trajectories/create/drawByParticleID ecalPID",
        "/vis/modeling/trajectories/ecalPID/default/setDrawStepPts false",
        f"/vis/modeling/trajectories/ecalPID/setDefault {DEFAULT_TRAJECTORY_COLOUR}",
        *(f"/vis/modeling/trajectories/ecalPID/set {name} {colour}" for name, colour in TRAJECTORY_COLOURS.items()),
    ]
    if min_track_mev > 0:
        commands += [
            "/vis/filtering/trajectories/create/attributeFilter ecalMinEnergy",
            "/vis/filtering/trajectories/ecalMinEnergy/setAttribute IKE",
            f"/vis/filtering/trajectories/ecalMinEnergy/addInterval {min_track_mev} MeV 1000 TeV",
        ]
    commands += [
        "/vis/scene/endOfEventAction accumulate",
        "/vis/scene/add/axes -340 -340 -10 100 mm",
        "/vis/scene/add/scale 100 mm",
        "/vis/scene/add/eventID",
        "/vis/viewer/set/autoRefresh true",
    ]
    return commands


class EventDisplay:
    """The production ECALTransport plus a vis manager and a Python-side draw command."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pilot = load_pilot_config(args.config)
        if args.sample not in self.pilot.samples:
            raise PilotConfigError(f"sample {args.sample!r} not in {sorted(self.pilot.samples)}")
        self.spec = self.pilot.samples[args.sample]
        geometry = load_geometry(GEOMETRY_CONFIG)
        transport = load_transport_config(TRANSPORT_CONFIG)
        try:
            self.band = self.pilot.energy_band_gev(args.energy)
        except ValueError as error:
            raise PilotConfigError(str(error)) from error
        self._seed_stream = (
            iter(range(args.seed, args.seed + 10**9))
            if args.seed is not None
            else iter(event_seeds(self.spec.base_seed, args.energy, 1000))
        )
        # Exactly the construction run_batch uses.
        started = time.perf_counter()
        self.app = ECALTransport(**_setup(self.pilot, self.spec, geometry, transport))
        print(f"[event_display] ECALTransport built in {time.perf_counter() - started:.1f} s")
        g4 = self.app.g4
        self.g4 = g4
        self.ui = g4.G4UImanager.GetUIpointer()
        self.vis = g4.G4VisExecutive("quiet")
        self.vis.Initialize()
        self._messenger = self._make_messenger()

    # --- vis -------------------------------------------------------------

    def open_viewer(self) -> None:
        offscreen = Path(self.args.offscreen).resolve() if self.args.offscreen else None
        if offscreen is not None:
            offscreen.parent.mkdir(parents=True, exist_ok=True)
        for command in vis_commands(
            driver=self.args.driver,
            fibres=self.args.fibres == "on",
            offscreen=offscreen,
            size=self.args.size,
            min_track_mev=self.args.min_track_mev,
        ):
            _apply(self.ui, command)

    # --- events ----------------------------------------------------------

    def draw_events(self, seeds: list[int]) -> list[EventResult]:
        started = time.perf_counter()
        results = self.app.run(self.args.energy, list(range(len(seeds))), seeds, self.band)
        wall = time.perf_counter() - started
        for result in results:
            print(
                f"[event_display] seed {result.seed}: E = {result.primary_energy_gev:.2f} GeV, "
                f"theta = {math.degrees(result.primary_theta_rad):.2f} deg, "
                f"phi = {math.degrees(result.primary_phi_rad):.1f} deg, "
                f"entry (x, y) = ({result.entry_x_mm:.1f}, {result.entry_y_mm:.1f}) mm at z = 0, "
                f"{result.fibre_ids.size} fibres hit, "
                f"E_fibre = {float(result.fibre_edep_mev.sum()) / 1000:.2f} GeV"
            )
        print(f"[event_display] {len(seeds)} event(s) simulated and drawn in {wall:.1f} s")
        return results

    def next_seeds(self, count: int) -> list[int]:
        return [next(self._seed_stream) for _ in range(count)]

    def _make_messenger(self) -> object:
        """Register /ecal/drawEvent [seed]: simulate and draw one event with the production generator."""

        g4 = self.g4
        display = self

        class Messenger(g4.G4UImessenger):
            def __init__(self) -> None:
                super().__init__()
                self.directory = g4.G4UIdirectory("/ecal/")
                self.directory.SetGuidance("AMS ECAL event display commands")
                self.draw = g4.G4UIcmdWithAnInteger("/ecal/drawEvent", self)
                self.draw.SetGuidance(
                    "Simulate and draw one event. Argument: event seed (0 = next seed of the configured stream). "
                    "Use this instead of /run/beamOn, which has no seed list."
                )
                self.draw.SetParameterName("seed", True)
                self.draw.SetDefaultValue(0)

            def SetNewValue(self, command, value):
                try:
                    seed = int(value) if value else 0
                    display.draw_events([seed] if seed else display.next_seeds(1))
                except BaseException:  # noqa: BLE001 - never let an exception cross into Geant4
                    import traceback

                    traceback.print_exc()

        return Messenger()

    # --- sessions --------------------------------------------------------

    def interactive(self) -> None:
        argv = ["event_display"]
        try:
            session = self.g4.G4UIExecutive(len(argv), argv, "win32")
        except Exception:  # noqa: BLE001
            session = self.g4.G4UIExecutive(len(argv), argv)
        self.open_viewer()
        self.draw_events(self.next_seeds(self.args.events))
        print(
            "[event_display] Window open. Mouse: left-drag rotates, right-drag/wheel zooms. "
            "Type /ecal/drawEvent [seed] for another event; exit the window to quit."
        )
        session.SessionStart()

    def offscreen(self) -> Path:
        self.open_viewer()
        self.draw_events(self.next_seeds(self.args.events))
        # Make sure the buffered scene is written for the file driver.
        _apply(self.ui, "/vis/viewer/rebuild", strict=False)
        _apply(self.ui, "/vis/viewer/flush", strict=False)
        return Path(self.args.offscreen).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Geant4 event display for the AMS ECAL.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="pilot YAML (default: realistic electron)")
    parser.add_argument("--sample", default="pilot")
    parser.add_argument(
        "--energy",
        type=float,
        default=50.0,
        help="GeV; for banded configs the band's lower edge (batch key), for fixed configs the exact energy",
    )
    parser.add_argument("--seed", type=int, help="first event seed (default: the configured seed stream)")
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--driver", choices=["TSG", "OGL"], default="TSG")
    parser.add_argument("--offscreen", metavar="OUT.PNG", help="render to a PNG with TSG_OFFSCREEN, no window")
    parser.add_argument("--fibres", choices=["on", "off"], default="on", help="draw the ~43,000 fibre volumes")
    parser.add_argument("--size", type=int, default=1000, help="viewer size in pixels (square)")
    parser.add_argument(
        "--min-track-mev",
        type=float,
        default=0.0,
        help="hide trajectories with initial kinetic energy below this (MeV); 0 draws all",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.energy > WARN_ABOVE_GEV:
        print(
            f"[event_display] WARNING: {args.energy:g} GeV showers have a very large trajectory count; "
            "drawing may be slow or exhaust memory. Consider --min-track-mev 5.",
            file=sys.stderr,
        )
    display = EventDisplay(args)
    if args.offscreen:
        print(f"[event_display] wrote {display.offscreen()}")
    else:
        display.interactive()


if __name__ == "__main__":
    main()
