# primer_dia_en_la_oficina
un pequeño proyecto que va a crecer

## App Flask de optimizacion lineal

La aplicacion en `app.py` construye un problema de programacion lineal entera usando los CSV del dataset de batallas.

### Fuente de datos

Busca los datos en este orden:

1. Variable de entorno `ARCHIVE_DIR`
2. Carpeta local `archive/`
3. `~/Downloads/archive`
4. `data/` dentro del repositorio

### Ejecutar en local

1. Instalar Python 3.11+
2. Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

3. Opcional: fijar carpeta `archive` explicita:

```powershell
$env:ARCHIVE_DIR="C:\Users\a8t\Downloads\archive"
```

4. Lanzar Flask:

```powershell
python app.py
```

5. Abrir en navegador:

`http://127.0.0.1:5000`
