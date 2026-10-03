# Windows equivalent of paper/build.sh: regenerate numbers, tables and figures from results/,
# then build the PDF with MiKTeX (or any TeX distribution on PATH).
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File paper\build.ps1
# Native tools (MiKTeX in particular) write notices to stderr; judge success by exit codes only.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
python -W ignore analysis\make_paper_numbers.py; if ($LASTEXITCODE -ne 0) { exit 1 }
python -W ignore analysis\make_figures.py;      if ($LASTEXITCODE -ne 0) { exit 1 }
Set-Location (Join-Path $repo "paper")
$log = "build.log"
function Step($exe, $arg) {
    $out = & $exe @arg 2>&1 | Out-String
    Add-Content -Path $log -Value $out
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 30; Write-Output "FAILED: $exe $arg"; exit 1 }
    return $out
}
Set-Content -Path $log -Value ""
$null = Step pdflatex @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
$null = Step bibtex @("paper")
$null = Step pdflatex @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
$last = Step pdflatex @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
# Report the FINAL pass only; earlier passes legitimately show unresolved citations.
$last -split "`r?`n" | Select-String -Pattern "Warning|Overfull \\hbox \([0-9]{2,}|Output written" | ForEach-Object { $_.Line }
