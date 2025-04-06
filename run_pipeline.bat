@echo off

echo [1/3] Running fetch_pubmed.py...
python fetch_pubmed.py --config config.yaml
if %errorlevel% neq 0 (
    echo WARNING: fetch_pubmed.py finished with error code %errorlevel%. Continuing...
) else (
    echo fetch_pubmed.py completed successfully.
)

echo [2/3] Running generate_queries.py...
python generate_queries.py --config config.yaml
if %errorlevel% neq 0 (
    echo WARNING: generate_queries.py finished with error code %errorlevel%. Continuing...
) else (
    echo generate_queries.py completed successfully.
)

echo [3/3] Running analyze_and_improve_queries.py...
python analyze_and_improve_queries.py --config config.yaml
if %errorlevel% neq 0 (
    echo WARNING: analyze_and_improve_queries.py finished with error code %errorlevel%. Continuing...
) else (
    echo analyze_and_improve_queries.py completed successfully.
)

echo Pipeline finished attempting all steps. 