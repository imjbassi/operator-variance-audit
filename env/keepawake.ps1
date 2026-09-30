# Windows-side watchdog for the WSL2 training grid. While it runs it:
#   1. prevents the machine from sleeping (SetThreadExecutionState; no system setting is changed,
#      the effect ends when this process exits; the display may still turn off),
#   2. holds a WSL session open so the VM idle timeout never fires,
#   3. every 2 minutes checks whether any queue worker is alive inside WSL; if none is and the
#      grid is incomplete, relaunches the workers (the queue is idempotent and skips finished runs).
# Log: %TEMP%\ova_keepawake.log
# Stop:  Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
#        Where-Object { $_.CommandLine -like '*keepawake.ps1*' } | ForEach-Object { Stop-Process -Id $_.ProcessId }
param([int]$TotalRuns = 138, [string]$Arms = "A B C", [int]$Workers = 3)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$log = Join-Path $env:TEMP "ova_keepawake.log"
$launch = "/mnt/c/Users/jaive.DESKTOP-3TNM9JL/Desktop/operator-variance-audit/env/launch_workers.sh"
function Log($m) { "$(Get-Date -Format s) $m" | Out-File -Append -FilePath $log }
Log "watchdog started pid $PID (TotalRuns=$TotalRuns Arms='$Arms' Workers=$Workers)"

$tick = 0
while ($true) {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    $w = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Where-Object { $_.CommandLine -like '*sleep infinity*' }
    if (-not $w) {
        Start-Process -FilePath "wsl.exe" -ArgumentList "-e","sleep","infinity" -WindowStyle Hidden
        Log "(re)started wsl keepalive session"
    }
    if ($tick % 2 -eq 0) {
        try {
            $alive = (wsl.exe -e bash -c "pgrep -fc scripts/run_queue.py" 2>$null | Out-String).Trim()
            $done  = (wsl.exe -e bash -c "find ~/ova/runs -name metrics.json -path '*/eval/*' | wc -l" 2>$null | Out-String).Trim()
            if (-not $alive) { $alive = "0" }
            if ([int]$alive -eq 0 -and [int]$done -lt $TotalRuns) {
                Log "no workers alive, $done/$TotalRuns done -> relaunching"
                $out = wsl.exe -e bash -c "ARMS='$Arms' NW=$Workers bash $launch" 2>&1 | Out-String
                Log $out.Trim()
            } elseif ($tick % 30 -eq 0) {
                Log "ok: workers=$alive done=$done/$TotalRuns"
            }
        } catch { Log "check failed: $_" }
    }
    $tick++
    Start-Sleep -Seconds 60
}
