$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$repoUrl = (git config --get remote.origin.url).Trim()
if (-not $repoUrl) {
    throw "Не настроен git remote origin."
}

$gitAuthHeader = ""
try {
    $ghToken = (gh auth token 2>$null).Trim()
    if ($ghToken) {
        $gitAuthHeader = "Authorization: Bearer $ghToken"
    }
} catch {
    $gitAuthHeader = ""
}

if (-not $gitAuthHeader) {
    try {
        $originUri = [Uri]$repoUrl
        $credentialInput = "protocol=$($originUri.Scheme)`nhost=$($originUri.Host)`n`n"
        $credentialLines = $credentialInput | git credential fill 2>$null
        $credential = @{}
        foreach ($line in $credentialLines) {
            $parts = $line -split "=", 2
            if ($parts.Count -eq 2) {
                $credential[$parts[0]] = $parts[1]
            }
        }
        if ($credential.username -and $credential.password) {
            $basicToken = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("$($credential.username):$($credential.password)"))
            $gitAuthHeader = "Authorization: Basic $basicToken"
        }
    } catch {
        $gitAuthHeader = ""
    }
}

$gitAuthSetup = "GIT_OPTS=()"
if ($gitAuthHeader) {
    $escapedGitAuthHeader = $gitAuthHeader.Replace("'", "'\''")
    $gitAuthSetup = "GIT_AUTH_HEADER='$escapedGitAuthHeader'`nGIT_OPTS=(-c `"http.extraHeader=`$GIT_AUTH_HEADER`")"
}

$sshKey = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
$remoteScript = @"
set -e
$gitAuthSetup
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq git >/dev/null 2>&1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq ffmpeg >/dev/null 2>&1
fi
mkdir -p /workspace
if [ ! -d /workspace/watermark/.git ]; then
  rm -rf /workspace/watermark
  git "`${GIT_OPTS[@]}" clone --branch main $repoUrl /workspace/watermark
else
  cd /workspace/watermark
  git remote set-url origin $repoUrl
  git "`${GIT_OPTS[@]}" fetch origin main
  if git show-ref --verify --quiet refs/heads/main; then
    git checkout main
  else
    git checkout -b main origin/main
  fi
  git "`${GIT_OPTS[@]}" pull --ff-only origin main
fi
cd /workspace/watermark
pip install -q -r requirements_web.txt
bash scripts/setup_propainter_runtime.sh
pkill -f uvicorn || true
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo [Успех] Сервер обновлен и работает в фоне на порту 8000!
"@

$remoteScript | & ssh -p 22115 -i $sshKey -o StrictHostKeyChecking=no root@69.30.85.178 "bash -s"
exit $LASTEXITCODE
