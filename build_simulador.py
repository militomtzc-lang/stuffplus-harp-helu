"""build_simulador.py — arma el simulador Harp Helú (un solo HTML autocontenido).
 
Para notebook, sin argparse:
    from build_simulador import build
    build()                                   # usa las rutas de abajo
    build(json_path="otra_malla.json")        # o cambia solo lo que necesites
 
Qué hace: lee la malla (dashboard_grid_v2.json), quita la columna n_support (el simulador no la usa),
y la incrusta en la plantilla en el marcador /*DATA*/ para producir simulador.html.
 
Cómo se genera la malla: celdas de notebook de la sesión del 9 oct (build_grid2, export a JSON).
Si cambias el modelo, regenera la malla y vuelve a correr build(); la plantilla no se toca.
Para editar el diseño o la lógica de la página, edita simulador_template.html (HTML + CSS + JS en un solo
archivo). La lógica vive en el bloque <script> final; las constantes de transporte vienen de meta.SHAPE del JSON.
"""
import json
from pathlib import Path
 
JSON_PATH = Path("resultados_stuffplus/retrain_oct2026/dashboard_grid_v2.json")
TEMPLATE_PATH = Path("simulador_template.html")
OUT_PATH = Path("simulador.html")
MARKER = "/*DATA*/"
 
 
def build(json_path=JSON_PATH, template_path=TEMPLATE_PATH, out_path=OUT_PATH):
    data = json.loads(Path(json_path).read_text())
    for g in data["grids"].values():
        g.pop("n_support", None)                       # solo sirve para diagnóstico en notebook
    blob = json.dumps(data, separators=(",", ":"))
    assert "</script" not in blob, "El JSON contiene '</script'; rompería la página."
    tpl = Path(template_path).read_text()
    assert tpl.count(MARKER) == 1, f"La plantilla debe tener exactamente un {MARKER}"
    out = Path(out_path)
    out.write_text(tpl.replace(MARKER, blob))
    print(f"{out} escrito: {out.stat().st_size / 1e6:.2f} MB, {len(data['grids'])} mallas (tipo x mano)")
    return out
 
 
if __name__ == "__main__":
    build()
