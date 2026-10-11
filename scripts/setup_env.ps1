# Install editable package + test runner (run from gov_form_assistant folder).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $root "..")

foreach ($venv in @(".venv", ".venv-gpu")) {
    $python = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $python) {
        Write-Host "Installing into $venv ..."
        & $python -m pip install -e . pytest -q
    }
}
Write-Host "Done. Use: .\.venv-gpu\Scripts\python.exe -m pytest tests"
