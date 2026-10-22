# Stuff+ calibrado para el Harp Helú
Hackathon Diablos Rojos, ISAC 2026.

- **Reporte técnico:** docs/Reporte_Tecnico_StuffPlus_HarpHelu.pdf
- **Simulador (abrir en Chrome):** https://militomtzc-lang.github.io/stuffplus-harp-helu/

## Contenido del repositorio
- `*.py`: pipeline (features, entrenamiento LightGBM, validación, simulación de altitud, construcción del simulador).
- `docs/`: reporte y simulador.

## Datos
Los datos pitch-by-pitch son de ISAC y no se redistribuyen.

## Herramientas y transparencia
Python, LightGBM, scikit-learn, pandas, NumPy, matplotlib. Se usó IA generativa (Claude, de Anthropic) como apoyo en código, revisión y redacción; las decisiones y cifras fueron revisadas por el equipo.

## Contenido
- `movement.py`, `physics.py`, `train_movement_models.py`: física y modelos de movimiento.
- `modeling.py`, `train_stuffplus.py`: entrenamiento y validación de Stuff+, Location+ y Pitching+ (LightGBM), versión inicial.
- `stuffplus_retrain.py`: entrenamiento final (objetivo v4), con el que se generan los resultados del reporte.
- `simulate_altitude.py`: contrafactual de altitud.
- `tunnel_features_fix.py`, `fix_time_to_plate_y_diagnostico_lateral_x.py`: variables de tunneling y correcciones del feed.
- `residual_profiles_k_por_tipo.py`: perfiles por tipo de pitch.
- `celdas_arsenal_ABC.py`, `celdas_shap_y_figura_caso.py`, `celda_figura_caso_v3.py`: análisis de arsenal, SHAP y figura del caso.
- `plot_arsenal.py`, `plot_physics_results.py`: figuras.
- `build_simulador.py`, `simulador_template.html`, `index.html`: simulador.
- `stuffplus_core.py`, `stuffplus_eval.py`: utilidades de features de tunneling y verificación de fuentes.
- `dashboard_grid_v2.json`: malla agregada que usa `build_simulador.py` para generar el simulador.
> Nota: algunos scripts importan `stuffplus_core` y `stuffplus_eval` (utilidades internas del equipo), que no se incluyen en este repositorio. El simulador publicado (`index.html`) ya incluye incrustada la malla de resultados.

