# Pre-interview validation (PowerShell)
# Usage: .\scripts\validate_deploy.ps1 https://your-app.example.com

param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$BaseUrl = $BaseUrl.TrimEnd("/")

Write-Host "== presentation UI (/) =="
try {
    $html = Invoke-WebRequest -Uri "$BaseUrl/" -UseBasicParsing
    if ($html.StatusCode -ne 200) { throw "GET / returned $($html.StatusCode)" }
    if ($html.Content -notmatch "JanusAI|root") {
        Write-Warning "GET / did not look like the Vite SPA — run: cd web && npm run build"
    } else {
        Write-Host "UI OK (status $($html.StatusCode))"
    }
} catch {
    Write-Warning "UI check failed: $_"
}

Write-Host "== health =="
Invoke-RestMethod "$BaseUrl/health" | ConvertTo-Json -Depth 6

Write-Host "== session =="
$session = Invoke-RestMethod -Method Post "$BaseUrl/sessions"
$sid = $session.session_id
Write-Host "session_id=$sid"

Write-Host "== ask (grounded) =="
$body = @{ question = "What arguments does HTTPException take?"; session_id = $sid } | ConvertTo-Json
Invoke-RestMethod -Method Post "$BaseUrl/ask" -ContentType "application/json" -Body $body |
    Select-Object request_id, cache_hit, intent, num_chunks, model_status |
    ConvertTo-Json

Write-Host "== six demo scenarios (API) =="
$scenarios = @(
    @{ name = "grounded"; q = "What arguments does HTTPException take?"; new = $false },
    @{ name = "abstain"; q = "How do I configure Redis connection pooling in FastAPI?"; new = $true },
    @{ name = "followup"; q = "and what are its attributes?"; new = $false },
    @{ name = "topic"; q = "How do I use UploadFile?"; new = $false },
    @{ name = "multi"; q = "What args does HTTPException take and how do I use UploadFile?"; new = $true },
    @{ name = "cache"; q = "What arguments does HTTPException take?"; new = $false }
)
foreach ($s in $scenarios) {
    if ($s.new) {
        $session = Invoke-RestMethod -Method Post "$BaseUrl/sessions"
        $sid = $session.session_id
    }
    $body = @{ question = $s.q; session_id = $sid } | ConvertTo-Json
    $r = Invoke-RestMethod -Method Post "$BaseUrl/ask" -ContentType "application/json" -Body $body
    Write-Host ("  {0}: cache_hit={1} intent={2} chunks={3} status={4}" -f $s.name, $r.cache_hit, $r.intent, $r.num_chunks, $r.model_status)
}

Write-Host "== eval runner =="
python -m evaluation.runner --base-url $BaseUrl --output evaluation/results/deploy_latest.json
Write-Host "OK: validated $BaseUrl"
Write-Host "Reminder: record docs/DEMO_SCRIPT.md walkthrough as interview fallback."
