"""Equations of the November 2026 group talk, typeset by LaTeX and exported as SVG (glyphs as paths, no font needed).

Each equation follows the code that produced the numbers on the slides:
- generalized dimensions: ams_ecal.multiscale_estimators.multiscale (partition_value, _dimension_from_slope,
  fitted_dimension: least-squares slope over a window of box sides s);
- AMS energy resolution: Kounine et al., NIM A 869 (2017) 110, Fig. 14, sigma = sqrt(10.5^2/E + 1.4^2) % (quadrature);
- longitudinal shower maximum: t_max = ln(E/E_c) + C_e with C_e = -0.5 for electrons (PDG, passage of particles
  through matter), in radiation lengths;
- proton survival: exp(-0.618) from our geometry's material table (presentation_figures_2026_11.PREFIX_LAMBDA_I);
- fibre saturation: ams_ecal.detector.fibre_saturation_calibration model M1, e' = e / (1 + e / e_s);
- photoelectrons: Poisson smearing with N p.e. per MIP-equivalent fibre energy (presentation_figures_2026_11.fig_resolution);
- EXP-019: Delta = L(C+M) - L(C), class-balanced log-loss, smallest gain of interest delta = 0.010.

    python -m ams_ecal.geant4_simulation.presentation_equations_2026_11
writes results/presentation_2026_11/equations/<name>.svg and sizes.json (width, height in pt). Needs latex and dvisvgm.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

OUT = Path("results/presentation_2026_11/equations")
INK = "101223"

EQUATIONS = {
    "partition": r"Z_q(s) \;=\; \sum_i p_i^{\,q} \;\propto\; s^{(q-1)\,D_q},\qquad p_i = \frac{E_i}{\sum_j E_j}",
    "entropy": r"S(s) \;=\; -\sum_i p_i \ln p_i \;\simeq\; -D_1 \ln s + \mathrm{const}",
    "slope": r"D_q = \frac{1}{q-1}\,\frac{\Delta \ln Z_q}{\Delta \ln s}\;\;(q \neq 1),\qquad D_1 = -\frac{\Delta S}{\Delta \ln s}",
    "ordering": r"D_0 \;\geq\; D_1 \;\geq\; D_2",
    "tmax": r"t_{\max} \;\approx\; \ln\frac{E}{E_c} - 0.5 \qquad [\,X_0\,]",
    "resolution": r"\frac{\sigma_E}{E} \;=\; \frac{10.5\%}{\sqrt{E/\mathrm{GeV}}} \,\oplus\, 1.4\% \;\equiv\; "
                  r"\sqrt{\frac{(10.5\%)^2}{E/\mathrm{GeV}} + (1.4\%)^2}",
    "survival": r"P_{\mathrm{no\,int}} \;=\; e^{-L/\lambda_I} \;=\; e^{-0.618} \;=\; 0.539",
    "saturation": r"e' \;=\; \frac{e}{1 + e/e_s},\qquad e_s = 260^{+107}_{-68}\ \mathrm{MeV}",
    "photoelectrons": r"n_{\mathrm{pe}} \sim \mathrm{Poisson}\!\left(N_{\mathrm{pe}}\,\frac{e}{e_{\mathrm{MIP}}}\right),\qquad N_{\mathrm{pe}} = 10",
    "delta": r"\Delta \;=\; \mathcal{L}_{C+M} - \mathcal{L}_{C},\qquad \delta = 0.010",
}

TEMPLATE = r"""\documentclass[border=1pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\begin{document}
\color[HTML]{%s}$\displaystyle %s$
\end{document}
"""


def render(name: str, tex: str, out: Path) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{name}.tex"
        src.write_text(TEMPLATE % (INK, tex), encoding="utf-8")
        subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", src.name], cwd=tmp, check=True,
                       capture_output=True)
        subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "-o", str((out / f"{name}.svg").resolve()), f"{name}.dvi"],
                       cwd=tmp, check=True, capture_output=True)
    svg = (out / f"{name}.svg").read_text(encoding="utf-8")
    width, height = (float(v) for v in re.search(r"width='([\d.]+)pt' height='([\d.]+)pt'", svg).groups())
    return width, height


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = {name: render(name, tex, OUT) for name, tex in EQUATIONS.items()}
    (OUT / "sizes.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    print(json.dumps(sizes, indent=1))


if __name__ == "__main__":
    main()
