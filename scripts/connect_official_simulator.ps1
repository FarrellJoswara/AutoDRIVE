param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,

    [string]$IpAddress = "127.0.0.1",

    [ValidateRange(1, 65535)]
    [int]$Port = 4567,

    [ValidateRange(1, 120)]
    [int]$ConnectTimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This helper controls the official Windows simulator UI and must run on Windows."
}

$process = Get-Process -Id $ProcessId
if ($process.ProcessName -ne "AutoDRIVE Simulator") {
    throw "PID $ProcessId is '$($process.ProcessName)', not AutoDRIVE Simulator."
}
if ($process.MainWindowHandle -eq 0) {
    throw "AutoDRIVE Simulator PID $ProcessId does not have a targetable window."
}

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class AutoDriveWindowInput {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr hWnd, ref POINT point);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extraInfo);
}
'@

$window = [IntPtr]$process.MainWindowHandle
$clientRect = New-Object AutoDriveWindowInput+RECT
if (-not [AutoDriveWindowInput]::GetClientRect($window, [ref]$clientRect)) {
    throw "Could not read the simulator client area."
}
$width = $clientRect.Right - $clientRect.Left
$height = $clientRect.Bottom - $clientRect.Top
if ($width -lt 900 -or $height -lt 500) {
    throw "Simulator window is too small for safe Menu Panel targeting ($width x $height)."
}

# The official Menu Panel is a Unity canvas, so its buttons are not exposed by
# Windows UI Automation. Fail closed unless the white Connection Button icon is
# visible at the documented panel position.
$origin = New-Object AutoDriveWindowInput+POINT
$origin.X = 0
$origin.Y = 0
if (-not [AutoDriveWindowInput]::ClientToScreen($window, [ref]$origin)) {
    throw "Could not resolve the simulator client area on screen."
}

function Invoke-CanvasClick([double]$xFraction, [double]$yFraction) {
    $x = $origin.X + [int][Math]::Round($width * $xFraction)
    $y = $origin.Y + [int][Math]::Round($height * $yFraction)
    [void][AutoDriveWindowInput]::SetCursorPos($x, $y)
    Start-Sleep -Milliseconds 80
    [AutoDriveWindowInput]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 60
    [AutoDriveWindowInput]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 120
}

function Get-ConnectionButtonBrightness {
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($origin.X, $origin.Y, 0, 0, [System.Drawing.Size]::new($width, $height))
    $centerX = [int][Math]::Round($width * 0.04)
    $centerY = [int][Math]::Round($height * 0.377)
    $count = 0
    for ($dy = -16; $dy -le 16; $dy++) {
        for ($dx = -16; $dx -le 16; $dx++) {
            if (($dx * $dx + $dy * $dy) -gt 256) { continue }
            $pixel = $bitmap.GetPixel($centerX + $dx, $centerY + $dy)
            if ($pixel.R -ge 210 -and $pixel.G -ge 210 -and $pixel.B -ge 210) {
                $count++
            }
        }
    }
    $graphics.Dispose()
    $bitmap.Dispose()
    return $count
}

$brightPixels = Get-ConnectionButtonBrightness
if ($brightPixels -lt 300) {
    # The simulator starts with its panels collapsed. Open the left Menu Panel,
    # then re-check the button before sending any form input or connection click.
    Invoke-CanvasClick 0.04 0.04
    Start-Sleep -Milliseconds 500
    $brightPixels = Get-ConnectionButtonBrightness
}
if ($brightPixels -lt 300) {
    throw "Connection Button was not detected. Open the simulator Menu Panel and keep the window visible; no click was sent."
}

# A live TCP connection to the selected port means this simulator is already
# connected. This keeps repeated startup checks from toggling or disturbing it.
$existingConnection = Get-NetTCPConnection -OwningProcess $ProcessId -State Established -ErrorAction SilentlyContinue |
    Where-Object { $_.RemotePort -eq $Port } |
    Select-Object -First 1
if ($existingConnection) {
    Write-Output "Simulator PID $ProcessId already has an established connection to port $Port."
    exit 0
}

[void][AutoDriveWindowInput]::SetForegroundWindow($window)
Start-Sleep -Milliseconds 250

Invoke-CanvasClick 0.10 0.168
[System.Windows.Forms.SendKeys]::SendWait("^a")
[System.Windows.Forms.SendKeys]::SendWait($IpAddress)
Invoke-CanvasClick 0.10 0.283
[System.Windows.Forms.SendKeys]::SendWait("^a")
[System.Windows.Forms.SendKeys]::SendWait([string]$Port)
Invoke-CanvasClick 0.10 0.377

$deadline = (Get-Date).AddSeconds($ConnectTimeoutSeconds)
do {
    $connection = Get-NetTCPConnection -OwningProcess $ProcessId -State Established -ErrorAction SilentlyContinue |
        Where-Object { $_.RemotePort -eq $Port } |
        Select-Object -First 1
    if ($connection) {
        Write-Output "Official simulator PID $ProcessId connected to $IpAddress`:$Port."
        exit 0
    }
    Start-Sleep -Milliseconds 250
} while ((Get-Date) -lt $deadline)

throw "Clicked the Connection Button but no TCP connection to $IpAddress`:$Port appeared within $ConnectTimeoutSeconds seconds. Check the bridge listener and simulator status."
