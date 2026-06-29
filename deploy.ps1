# deploy.ps1
param (
    [ValidateSet("Lab", "TSUBAME")]
    [string]$Target = "TSUBAME"
)

$sourcePath = "./"
$rsyncExe   = "C:/Program Files/Git/usr/bin/rsync.exe"
$sshExe     = "C:/Program Files/Git/usr/bin/ssh.exe"

if (-Not (Test-Path $rsyncExe)) {
    Write-Host "Error: rsync.exe not found at $rsyncExe." -ForegroundColor Red
    exit 1
}

if ($Target -eq "Lab") {
    $destRemote = "mikael@172.20.25.11"
    $destPath   = "/data/mikael/LBM_particle_test/"
    $sshCmd = "'$sshExe' -F /dev/null -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedKeyTypes=+ssh-rsa"
    Write-Host "Target: KANDA-LAB (Legacy SSH)" -ForegroundColor Green
}
elseif ($Target -eq "TSUBAME") {
    $destRemote = "us06074@login.t4.gsic.titech.ac.jp" 
    $destPath   = "/gs/bs/tga-lbmcity/mikael/LBM_particle_test/"
    $sshCmd = "'$sshExe'"
    Write-Host "Target: TSUBAME 4.0" -ForegroundColor Green
}

$rsyncArgs = @(
    "-avPz",
    "-e", $sshCmd,
    "--exclude=.git/",
    "--exclude=deploy.ps1",
    "--exclude=Output_*/",
    "--exclude=*.exe",
    "--exclude=*.obj",
    "--exclude=Particle_PostProcess_Outputs",
    $sourcePath,
    "${destRemote}:${destPath}"
)

Write-Host "Deploying code via rsync..." -ForegroundColor Cyan
& ${rsyncExe} ${rsyncArgs}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Deployment failed. Check SSH/rsync configuration." -ForegroundColor Red
    exit 1
}

Write-Host "Deployment to $Target complete." -ForegroundColor Green