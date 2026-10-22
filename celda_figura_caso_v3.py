# celda_figura_caso_v3.py — 14 oct 2026. Reemplaza F1 v2. Mismos requisitos: raw_isac, normalize_types, build_primary_tunnel_features.
# Correcciones sobre v2 (vistas en la figura real):
#  1) Los 3 paneles usan la MISMA escala (misma ventana en pies): así se ve que la separación crece de verdad.
#  2) Se dibuja la dispersión de los pitches individuales (cruz = ±1 DE) de la primaria y del tipo foco:
#     una separación de promedios solo es "túnel" si es chica frente a esa dispersión.
#  3) Etiquetas en columna a la derecha con línea guía (antes se encimaban Curveball/Changeup/Splitter en el plato).
#  4) Se quitaron las líneas punteadas decisión→plato: salían del panel sin marcar el punto de origen.
#  5) Título neutral (antes: "de la Sinker").
import numpy as np, pandas as pd, matplotlib.pyplot as plt
 
PIT, YR, FOCO, MIN_N = "pitcher_00841", 2025, "Splitter", 50
INK, MUTED, GRID, BLUE, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#9a9992"
 
sub = normalize_types(raw_isac.loc[(raw_isac["pitcher_anon_id"] == PIT) & (raw_isac["year"] == YR)].copy())
L = build_primary_tunnel_features(sub)
if not L.index.equals(sub.index):
    assert len(L) == len(sub); L = L.set_axis(sub.index)
dfv = sub[["AutoPitchType"]].join(L)
cols = [f"{p}_{a}" for p in ("release", "decision", "plate") for a in ("x", "z")]
ok = dfv["has_reliable_anchor"].astype(bool) & dfv[cols].notna().all(axis=1)
g = dfv[ok].groupby("AutoPitchType", observed=True)
prof = g[cols].mean().join(g[cols].std().add_suffix("_sd")).join(g.size().rename("n"))
prof = prof[prof["n"] >= MIN_N]
prim = dfv["primary_pitch_type"].dropna().mode().iloc[0]
assert prim in prof.index and FOCO in prof.index
 
def dist_in(pt, a, b):
    return 12 * np.hypot(prof.loc[a, f"{pt}_x"] - prof.loc[b, f"{pt}_x"], prof.loc[a, f"{pt}_z"] - prof.loc[b, f"{pt}_z"])
def sd_in(pt, k):  # dispersión radial de los pitches individuales (1 DE por eje, combinada), en pulgadas
    return 12 * np.hypot(prof.loc[k, f"{pt}_x_sd"], prof.loc[k, f"{pt}_z_sd"])
d = {p: dist_in(p, FOCO, prim) for p in ("release", "decision", "plate")}
print("separación (in):", {k: round(v, 1) for k, v in d.items()})
for p in ("release", "decision", "plate"):
    print(f"dispersión 1 DE en {p} (in): {prim} {sd_in(p, prim):.1f} | {FOCO} {sd_in(p, FOCO):.1f}")
 
# ventana común en pies: la más grande de las tres etapas (rango de promedios) + holgura
span = 0
for p in ("release", "decision", "plate"):
    span = max(span, np.ptp(prof[f"{p}_x"]), np.ptp(prof[f"{p}_z"]))
span = span * 1.35 + 0.25
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stages = [("release", "Release"), ("decision", "Decisión (150 ms antes de que la recta llegue)"), ("plate", "Plato")]
for ax, (pt, ttl) in zip(axes, stages):
    cx, cz = prof[f"{pt}_x"].mean(), prof[f"{pt}_z"].mean()
    ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cz - span / 2, cz + span / 2)
    ax.set_aspect("equal", adjustable="box")
    for k in (prim, FOCO):  # dispersión de pitches individuales
        r = prof.loc[k]
        ax.errorbar(r[f"{pt}_x"], r[f"{pt}_z"], xerr=r[f"{pt}_x_sd"], yerr=r[f"{pt}_z_sd"],
                    fmt="none", ecolor=INK if k == prim else BLUE, elinewidth=1, alpha=0.35, capsize=0, zorder=2)
    pts = []
    for k, r in prof.iterrows():
        isprim, isfoco = k == prim, k == FOCO
        ax.scatter(r[f"{pt}_x"], r[f"{pt}_z"], s=110 if (isprim or isfoco) else 60, marker="s" if isprim else "o",
                   color=INK if isprim else (BLUE if isfoco else GRAY), edgecolor="white", linewidth=1.5, zorder=3)
        pts.append((k, r[f"{pt}_x"], r[f"{pt}_z"], int(r["n"])))
    gap, last = 0.075 * span, None   # etiquetas en columna a la derecha, sin encimarse
    for k, x, z, n in sorted(pts, key=lambda t: t[2]):
        ty = z if last is None else max(z, last + gap); last = ty
        ax.annotate(f"{k} (n={n})", (x, z), xytext=(1.04, (ty - (cz - span / 2)) / span),
                    textcoords=("axes fraction", "axes fraction"), fontsize=8, color=INK, va="center",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY, shrinkA=0, shrinkB=3))
    ax.set_title(ttl, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Lateral (ft)", color=MUTED, fontsize=9); ax.set_ylabel("Altura (ft)", color=MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8, zorder=0); ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
 
fig.suptitle(f"{PIT}, {YR}: separación {FOCO}–{prim} de {d['release']:.1f} in en el release, "
             f"{d['decision']:.1f} in en la decisión y {d['plate']:.1f} in en el plato",
             fontsize=11, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01,
         f"Misma escala en los tres paneles ({span:.1f} ft). Promedios por tipo (solo tipos con ≥{MIN_N} pitches y ancla confiable); "
         f"cruz = ±1 DE de los pitches individuales de {prim} y {FOCO}. Cuadrado = recta primaria. Asociativo y retrospectivo.",
         fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.05, right=0.93, wspace=0.75, top=0.86, bottom=0.14)
fig.savefig("fig_caso_release_decision_plato_v3.png", dpi=200, bbox_inches="tight")
print("guardada: fig_caso_release_decision_plato_v3.png")
plt.show()
 
