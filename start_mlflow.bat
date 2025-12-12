@echo off
REM ---------------------------
REM Start MLflow server (Windows) — fixed for paths with spaces
REM ---------------------------

REM Base folder (script folder)
set "BASE_DIR=%~dp0"

REM DB and artifact paths (absolute)
set "DB_PATH=%BASE_DIR%mlflow_registry.db"
set "ARTIFACT_ROOT=%BASE_DIR%mlruns"
set "PORT=5000"

REM Create artifact folder if missing
IF NOT EXIST "%ARTIFACT_ROOT%" (
    mkdir "%ARTIFACT_ROOT%"
)

echo.
echo Starting MLflow server...
echo Backend DB: %DB_PATH%
echo Artifact root: %ARTIFACT_ROOT%
echo Host: 0.0.0.0  Port: %PORT%
echo.

REM Convert backslashes to forward slashes for sqlite URI
REM (this avoids problems with 'sqlite:///' + backslashes and spaces)
set "DB_PATH_SLASH=%DB_PATH:\=/%"

REM Build quoted URI and quoted artifact root
set "BACKEND_URI=sqlite:///%DB_PATH_SLASH%"
set "QUOTED_BACKEND_URI="%BACKEND_URI%""
set "QUOTED_ARTIFACT_ROOT="%ARTIFACT_ROOT%""


REM Run mlflow server (arguments are quoted where needed)
mlflow server ^
  --backend-store-uri %QUOTED_BACKEND_URI% ^
  --default-artifact-root %QUOTED_ARTIFACT_ROOT% ^
  --host 0.0.0.0 ^
  --port %PORT%


start "" "http://localhost:%PORT%"

echo.
echo MLflow server stopped (or failed to start).
echo Try visiting: http://localhost:%PORT%
pause
