$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$repoUrl = (git config --get remote.origin.url).Trim()
if (-not $repoUrl) {
    throw "Git remote origin is not configured."
}

$sshKey = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
$remoteHost = "root@69.30.85.178"
$remotePort = 22115
$remoteBundle = "/tmp/watermark_deploy.bundle"
$localBundle = Join-Path $env:TEMP ("watermark_deploy_{0}.bundle" -f ([guid]::NewGuid().ToString("N")))

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

try {
    & git bundle create $localBundle HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create git bundle."
    }

    & scp -P $remotePort -i $sshKey -o StrictHostKeyChecking=no $localBundle "${remoteHost}:$remoteBundle"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload git bundle to RunPod."
    }

    $remoteScript = @"
set -e
$gitAuthSetup
REMOTE_BUNDLE='$remoteBundle'
REPO_URL='$repoUrl'
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
  mkdir -p /workspace/watermark
  cd /workspace/watermark
  git init
  git remote add origin "`$REPO_URL"
else
  cd /workspace/watermark
  git remote set-url origin "`$REPO_URL"
fi
if git "`${GIT_OPTS[@]}" fetch origin main; then
  echo '[deploy] fetched origin/main from GitHub'
else
  echo '[deploy] GitHub fetch failed, using uploaded bundle'
  git fetch "`$REMOTE_BUNDLE" main:refs/remotes/origin/main
fi
if git show-ref --verify --quiet refs/heads/main; then
  git checkout main
  git merge --ff-only origin/main || git checkout -B main origin/main
else
  git checkout -b main origin/main
fi
cd /workspace/watermark
pip install -q -r requirements_web.txt
bash scripts/setup_propainter_runtime.sh
pkill -f uvicorn || true
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo '[deploy] server updated and running on port 8000'
"@

    $remoteScript | & ssh -p $remotePort -i $sshKey -o StrictHostKeyChecking=no $remoteHost "bash -s"
    exit $LASTEXITCODE
} finally {
    if (Test-Path $localBundle) {
        Remove-Item -Force $localBundle
    }
}