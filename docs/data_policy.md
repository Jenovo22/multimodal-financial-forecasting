# Data Policy

The project uses market, macro and option-chain data. These files can grow quickly and may change often, so collaboration should be based on reproducible commands rather than committing new generated data.

## Versioned Data

Some historical data files are already tracked in Git. Treat them as legacy seed/reference data until the team decides to migrate them to external storage.

Do not remove or rewrite tracked data in the same PR as model/code changes unless the PR is explicitly a data migration.

## Generated Data

New generated data should stay local and is ignored by `.gitignore`:

- `Data/raw/`
- `Data/processed/`
- `Splits/`
- `artifacts/`
- `reports/`

If collaborators need the same generated file, document:

- Source command.
- Source date/time.
- Symbol/universe.
- Filters.
- Expected row count.
- Checksum if the file is large or shared externally.

## Particiones de datos en Splits/

`Data/` contiene los datos originales o base del proyecto. Debe seguir siendo la fuente primaria para reproducir pipelines, reconstruir datasets y auditar de donde salio cada archivo.

`Splits/` contiene particiones generadas para entrenamiento, validacion y prueba. Es una carpeta derivada: organiza copias o subconjuntos de archivos de `Data/`, pero no reemplaza a `Data/` ni debe tratarse como fuente de verdad.

La particion preserva la estructura original de carpetas. Un ejemplo de la estructura esperada es:

```text
Splits/
  expiration_split/
    train/
      Data/
        raw/
        processed/
        final/
    validation/
      Data/
        raw/
        processed/
        final/
    test/
      Data/
        raw/
        processed/
        final/
    shared/
    manifests/
```

`shared/` se usa para archivos que no se pueden partir directamente o que deben compartirse entre splits. Esto incluye archivos sin una columna reconocida para particion, archivos no CSV cuando se copian explicitamente, o filas sin clave de particion.

`manifests/` contiene archivos de auditoria generados por el script de particion, como `split_manifest.json` y `split_summary.csv`. Estos archivos ayudan a revisar que fuentes se usaron, cuantos registros terminaron en cada split, que columna se uso para partir y que rango de claves cubre cada salida.

Estrategia recomendada:

- `expiration`: recomendada con el estado actual, donde solo existe un snapshot de opciones. Separa por expiracion y evita mezclar vencimientos futuros dentro del entrenamiento principal.
- `timestamp`: recomendada cuando existan multiples snapshots diarios de opciones. Permite una evaluacion temporal mas honesta, entrenando con fechas antiguas y probando con fechas futuras no vistas.
- `random`: no debe usarse como particion principal para evaluacion seria. Solo tendria sentido para debugging o pruebas rapidas si existiera una variante del flujo que lo soporte.

`Splits/` ayuda a organizar y auditar particiones en disco, pero por si sola no mejora la generalizacion del modelo. La validacion temporal real sigue dependiendo de tener multiples snapshots diarios.

Los datos generados en `Splits/` no deberian commitearse salvo decision explicita del equipo.

## Model Artifacts

Checkpoints such as `artifacts/finn_model.pt` are local outputs. Do not commit them unless the team explicitly creates a release process for model artifacts.

Recommended pattern:

```text
artifacts/
  finn_model.pt
  finn_model_selection.pt
reports/
  finn_training_metrics.json
  prediction_dashboard.html
```

## Secrets

Do not commit `.env`, API keys, broker credentials or paid-data credentials. Use `.env.example` for non-secret defaults.

## Future Recommendation

If data size grows, introduce one of these:

- Cloud object storage for datasets and checkpoints.
- DVC for reproducible dataset versioning.
- GitHub Releases for frozen small artifacts.
