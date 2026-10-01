# Windows-side watchdog for long jobs running under nohup inside WSL2. While it runs it:
#   1. prevents the machine from sleeping (SetThreadExecutionState; no system setting is changed,
#      the effect ends when this process exits; the display may still turn off),
#   2. holds a WSL session open so the VM idle timeout never fires,
#   3. every 2 minutes checks whether any worker is alive inside WSL; if none is and the job is
#      incomplete, relaunches the workers (both queues are idempotent and skip finished work),
#   4. exits by itself once the job is complete.
# Background: on 2026-09-29/30 the Microsoft Store auto-updated the WSL package twice and the
# installer terminated the VM mid-run; this watchdog makes such a kill cost minutes, not hours.
# Log: %TEMP%\ova_keepawake.log
# Stop:  Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
#        Where-Object { $_.CommandLine -like '*keepawake.ps1*' } | ForEach-Object { Stop-Process -Id $_.ProcessId }
#
# Stage 1 (training grid):  -Mode train   (138 runs,  env/launch_workers.sh)
# Stage 2 (re-evaluation):  -Mode stage2  (460 chunks, env/launch_stage2.sh)
param([ValidateSet("train", "stage2")][string]$Mode = "train", [int]$Workers = 0)

$repo = "/mnt/c/Users/jaive.DESKTOP-3TNM9JL/Desktop/operator-variance-audit"
if ($Mode -eq "train") {
    $total = 138; if ($Workers -le 0) { $Workers = 3 }
    $aliveCmd = "pgrep -fc scripts/run_queue.py"
    $doneCmd  = "find ~/ova/runs -name metrics.json -path '*/eval/*' | wc -l"
    $launch   = "ARMS='A B C' NW=$Workers bash $repo/env/launch_workers.sh"
} else {
    $total = 460; if ($Workers -le 0) { $Workers = 8 }
    $aliveCmd = "pgrep -fc scripts/eval_stage2_queue.py"
    $doneCmd  = "find ~/ova/runs -name 'rollouts_*.csv' -path '*/eval_stage2/*' | wc -l"
    $launch   = "NW=$Workers bash $repo/env/launch_stage2.sh"
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$log = Join-Path $env:TEMP "ova_keepawake.log"
function Log($m) { "$(Get-Date -Format s) $m" | Out-File -Append -FilePath $log }
Log "watchdog started pid $PID (Mode=$Mode total=$total Workers=$Workers)"

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
            $alive = (wsl.exe -e bash -c $aliveCmd 2>$null | Out-String).Trim()
            $done  = (wsl.exe -e bash -c $doneCmd 2>$null | Out-String).Trim()
            if (-not $alive) { $alive = "0" }
            if ([int]$done -ge $total -and [int]$alive -eq 0) {
                Log "job complete ($done/$total); watchdog exiting and releasing keep-awake"
                Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Where-Object { $_.CommandLine -like '*sleep infinity*' } |
                    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
                break
            }
            if ([int]$alive -eq 0) {
                Log "no workers alive, $done/$total done -> relaunching"
                $out = wsl.exe -e bash -c $launch 2>&1 | Out-String
                Log $out.Trim()
            } elseif ($tick % 30 -eq 0) {
                Log "ok: workers=$alive done=$done/$total"
            }
        } catch { Log "check failed: $_" }
    }
    $tick++
    Start-Sleep -Seconds 60
}
