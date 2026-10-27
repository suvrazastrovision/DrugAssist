param(
    [ValidateRange(1024, 65535)][int]$ApiPort = 8000,
    [ValidateRange(1024, 65535)][int]$UiPort = 8501,
    [switch]$NoBrowser,
    [switch]$FromSource
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$backend = $null
$frontend = $null
$originalLocation = Get-Location
$previousEnvironment = @{}
foreach ($name in @('HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_HUB_OFFLINE',
                    'TRANSFORMERS_OFFLINE', 'DRUGASSIST_API_URL')) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Require-Success([string]$Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}

function Wait-Ready([string]$Url, $Process) {
    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) { throw 'A server exited. Check results/private/startup/ logs.' }
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw "Startup timed out: $Url. Check results/private/startup/ logs."
}

try {
    Set-Location -LiteralPath $root
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
    }
    if ($ApiPort -eq $UiPort) { throw 'Backend and frontend ports must be different.' }
    foreach ($port in @($ApiPort, $UiPort)) {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
        try { $listener.Start() }
        catch { throw "Port $port is in use. Stop the existing app, or choose -ApiPort / -UiPort." }
        finally { $listener.Stop() }
    }

    Write-Host 'Checking dependencies (first installation may take several minutes)...'
    # Preserve optional notebook/test packages already installed in this environment.
    & uv sync --locked --inexact
    Require-Success 'Dependency setup failed. Stop running notebooks/servers, then retry.'
    $python = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
    }
    & $python -c "from dotenv import load_dotenv; import os; load_dotenv('.env'); raise SystemExit(0 if os.getenv('OPENAI_API_KEY', '').strip() else 1)"
    if ($LASTEXITCODE -ne 0) {
        $secureKey = Read-Host 'Enter your OpenAI API key (saved only in your local .env)' -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        $previousKey = $env:OPENAI_API_KEY
        try {
            $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
            if (-not $env:OPENAI_API_KEY) { throw 'An API key is required for new questions and answers.' }
            & $python -c "from dotenv import set_key; import os; set_key('.env', 'OPENAI_API_KEY', os.environ['OPENAI_API_KEY'])"
            Require-Success 'Could not save the API key in .env.'
        } finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
            $env:OPENAI_API_KEY = $previousKey
            $secureKey = $null
        }
    }

    Write-Host 'Preparing or checking the local collection...'
    # Setup may download public assets; offline flags apply only after it completes.
    $env:HF_HUB_OFFLINE = '0'
    $env:TRANSFORMERS_OFFLINE = '0'
    if ($FromSource) {
        & $python -m src.cli setup --from-source
    } else {
        & $python -m src.cli setup
    }
    Require-Success 'Setup failed. See the message above; no paid fallback was attempted.'

    # Bundled model configuration and weights must use the same local HF cache.
    $env:HF_HUB_CACHE = Join-Path $root 'data\models'
    $env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:DRUGASSIST_API_URL = "http://127.0.0.1:$ApiPort"
    $logs = Join-Path $root 'results\private\startup'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $backend = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -ArgumentList @('-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', $ApiPort) `
        -RedirectStandardOutput (Join-Path $logs 'backend.log') `
        -RedirectStandardError (Join-Path $logs 'backend-error.log')
    Write-Host 'Loading the prepared search collection and local reranker...'
    Wait-Ready "http://127.0.0.1:$ApiPort/health" $backend
    $frontend = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -ArgumentList @('-m', 'streamlit', 'run', 'frontend/app.py', '--server.address', '127.0.0.1', `
            '--server.port', $UiPort, '--server.headless', 'true') `
        -RedirectStandardOutput (Join-Path $logs 'frontend.log') `
        -RedirectStandardError (Join-Path $logs 'frontend-error.log')
    Wait-Ready "http://127.0.0.1:$UiPort/_stcore/health" $frontend
    Write-Host "DrugAssist is ready: http://127.0.0.1:$UiPort"
    Write-Host 'Keep this terminal open. Press Ctrl+C to stop both servers.'
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$UiPort" }
    while (-not $backend.HasExited -and -not $frontend.HasExited) { Start-Sleep -Seconds 1 }
    throw 'A server stopped. Check results/private/startup/ logs.'
} catch {
    Write-Host "DrugAssist: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    foreach ($process in @($frontend, $backend)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    }
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
    Set-Location -LiteralPath $originalLocation.Path
}
