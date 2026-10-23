param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class LabWindow {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct Point { public int X, Y; }
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr window, ref Point point);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
}
'@
$artifactDir = Join-Path $ProjectRoot 'build-debug\verification'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
$process = Start-Process -FilePath (Join-Path $ProjectRoot 'build-debug\vulkan-starter-app.exe') -ArgumentList '--verify' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $artifactDir 'stdout.log') -RedirectStandardError (Join-Path $artifactDir 'stderr.log')
$process.Handle | Out-Null
$timer = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($capture in @(@{Time=700;Name='perspective'}, @{Time=2400;Name='orthographic'}, @{Time=4500;Name='paused'}, @{Time=6500;Name='uniform-color'})) {
    while ($timer.ElapsedMilliseconds -lt $capture.Time -and !$process.HasExited) { Start-Sleep -Milliseconds 30 }
    $process.Refresh()
    if ($process.HasExited) { break }
    $window = $process.MainWindowHandle
    if ($window -eq [IntPtr]::Zero) { continue }
    [LabWindow]::SetForegroundWindow($window) | Out-Null
    $rect = New-Object LabWindow+Rect
    $origin = New-Object LabWindow+Point
    [LabWindow]::GetClientRect($window, [ref]$rect) | Out-Null
    [LabWindow]::ClientToScreen($window, [ref]$origin) | Out-Null
    $bitmap = New-Object System.Drawing.Bitmap($rect.Right, $rect.Bottom)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($origin.X, $origin.Y, 0, 0, $bitmap.Size)
    $bitmap.Save((Join-Path $artifactDir ($capture.Name + '.png')))
    $coloredPixels = 0
    for ($x = 370; $x -lt $bitmap.Width; $x += 4) {
        for ($y = 20; $y -lt $bitmap.Height - 80; $y += 4) {
            $pixel = $bitmap.GetPixel($x, $y)
            if ([Math]::Abs([int]$pixel.R - [int]$pixel.G) -gt 35 -or
                [Math]::Abs([int]$pixel.R - [int]$pixel.B) -gt 35 -or
                [Math]::Abs([int]$pixel.G - [int]$pixel.B) -gt 35) { ++$coloredPixels }
        }
    }
    Write-Output ($capture.Name + ': ' + $coloredPixels + ' sampled object pixels')
    $graphics.Dispose()
    $bitmap.Dispose()
    if ($coloredPixels -lt 50) { throw ('No visible geometry in ' + $capture.Name) }
}
if (!$process.WaitForExit(15000)) { throw 'Verification did not finish within 15 seconds' }
if ($process.ExitCode -ne 0) { throw ('Application exited with ' + $process.ExitCode) }
$errors = Get-Content (Join-Path $artifactDir 'stderr.log') -Raw
$output = Get-Content (Join-Path $artifactDir 'stdout.log') -Raw
if ($errors -or $output -match 'Validation Error|VUID-|SYNC-HAZARD') {
    Write-Output $output
    Write-Output $errors
    throw 'Application or Vulkan validation reported an error'
}
Write-Output 'PASS: application exited normally; validation logs are empty.'
