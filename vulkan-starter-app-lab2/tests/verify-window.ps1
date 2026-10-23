param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$artifactDir = Join-Path $ProjectRoot 'build-debug\verification\gpu'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$stamp = [DateTime]::Now.ToString('yyyyMMdd-HHmmss-fff')
$stdout = Join-Path $artifactDir ($stamp + '-stdout.log')
$stderr = Join-Path $artifactDir ($stamp + '-stderr.log')
$env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
$started = Get-Date
$process = Start-Process -FilePath (Join-Path $ProjectRoot 'build-debug\vulkan-starter-app.exe') -ArgumentList '--verify' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Handle | Out-Null
try {
    if (!$process.WaitForExit(30000)) { throw 'Verification timed out' }
    if ($process.ExitCode -ne 0) { throw ('Application exited with ' + $process.ExitCode) }
    $output = Get-Content -LiteralPath $stdout -Raw
    $errors = Get-Content -LiteralPath $stderr -Raw
    if ($errors -or $output -match 'Validation Error|VUID-|SYNC-HAZARD|\[ERROR') {
        Write-Output $output; Write-Output $errors
        throw 'Application or Vulkan validation reported an error'
    }
    foreach ($stage in @('directional','point','spot','multiple','ambient','material','camera','markers')) {
        $path = Join-Path $artifactDir ($stage + '.bmp')
        if (!(Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).LastWriteTime -lt $started) { throw ('Missing fresh capture: ' + $stage) }
        $bitmap = New-Object System.Drawing.Bitmap($path)
        $bitmap.Save((Join-Path $artifactDir ($stage + '.png')),[System.Drawing.Imaging.ImageFormat]::Png)
        $background = $bitmap.GetPixel($bitmap.Width-10,10)
        $geometry = 0
        for ($x = 405; $x -lt $bitmap.Width-10; $x += 6) {
            for ($y = 20; $y -lt $bitmap.Height-80; $y += 6) {
                $pixel = $bitmap.GetPixel($x,$y)
                $difference = [Math]::Abs([int]$pixel.R-$background.R)+[Math]::Abs([int]$pixel.G-$background.G)+[Math]::Abs([int]$pixel.B-$background.B)
                if ($difference -gt 50) { ++$geometry }
            }
        }
        $bitmap.Dispose()
        Write-Output ($stage + ': ' + $geometry + ' sampled geometry pixels')
        if ($geometry -lt 30) { throw ('Blank scene in ' + $stage) }
    }
    Write-Output 'PASS: eight GPU captures; clean Vulkan synchronization validation; normal shutdown.'
    Write-Output $artifactDir
} finally {
    if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
}
