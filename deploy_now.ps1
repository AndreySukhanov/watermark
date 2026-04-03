$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$repoUrl = (git config --get remote.origin.url).Trim()
if (-not $repoUrl) {
    throw "Не настроен git remote origin."
}

$ghToken = (gh auth token).Trim()
if (-not $ghToken) {
    throw "Не удалось получить GitHub token через gh auth."
}

$sshKey = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
$remoteScript = @"
set -e
GIT_AUTH_HEADER='Authorization: Bearer $ghToken'
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq git >/dev/null 2>&1
fi
mkdir -p /workspace
if [ ! -d /workspace/watermark/.git ]; then
  rm -rf /workspace/watermark
  git -c http.extraHeader="$GIT_AUTH_HEADER" clone --branch main $repoUrl /workspace/watermark
else
  cd /workspace/watermark
  git remote set-url origin $repoUrl
  git -c http.extraHeader="$GIT_AUTH_HEADER" fetch origin main
  if git show-ref --verify --quiet refs/heads/main; then
    git checkout main
  else
    git checkout -b main origin/main
  fi
  git -c http.extraHeader="$GIT_AUTH_HEADER" pull --ff-only origin main
fi
cd /workspace/watermark
pip install -q -r requirements_web.txt
pkill -f uvicorn || true
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo [Успех] Сервер обновлен и работает в фоне на порту 8000!
"@

$remoteScript | & ssh -p 22115 -i $sshKey -o StrictHostKeyChecking=no root@69.30.85.178 "bash -s"
exit $LASTEXITCODE
