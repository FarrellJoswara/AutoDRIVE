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

# The official player accepts -ip/-port on startup and can connect before its
# Unity canvas has opened the Menu Panel. Check the bridge first: a live
# simulator must not be reported as a startup failure just because the optional
# UI form is collapsed or unavailable.
$existingConnection = Get-NetTCPConnection -OwningProcess $ProcessId -State Established -ErrorAction SilentlyContinue |
    Where-Object { $_.RemotePort -eq $Port } |
    Select-Object -First 1
if ($existingConnection) {
    Write-Output "Simulator PID $ProcessId already has an established connection to port $Port; skipped the connection panel."
    exit 0
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
    [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int command);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint attach, uint attachTo, bool attachState);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
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

function Set-TargetWindowForeground {
    # Unity launches from several processes; Windows may refuse the ordinary
    # foreground request after the first window. Temporarily attach this input
    # thread to the active window's thread so each target window can be focused.
    $currentThread = [AutoDriveWindowInput]::GetCurrentThreadId()
    $foreground = [AutoDriveWindowInput]::GetForegroundWindow()
    $foregroundProcess = [uint32]0
    $foregroundThread = if ($foreground -ne [IntPtr]::Zero) {
        [AutoDriveWindowInput]::GetWindowThreadProcessId($foreground, [ref]$foregroundProcess)
    } else { [uint32]0 }
    $attached = $false
    if ($foregroundThread -ne 0 -and $foregroundThread -ne $currentThread) {
        $attached = [AutoDriveWindowInput]::AttachThreadInput($currentThread, $foregroundThread, $true)
    }
    try {
        [void][AutoDriveWindowInput]::ShowWindow($window, 9) # SW_RESTORE
        [void][AutoDriveWindowInput]::BringWindowToTop($window)
        [void][AutoDriveWindowInput]::SetForegroundWindow($window)
        Start-Sleep -Milliseconds 300
        if ([AutoDriveWindowInput]::GetForegroundWindow() -ne $window) {
            [void][AutoDriveWindowInput]::SetForegroundWindow($window)
            Start-Sleep -Milliseconds 300
        }
        return ([AutoDriveWindowInput]::GetForegroundWindow() -eq $window)
    } finally {
        if ($attached) {
            [void][AutoDriveWindowInput]::AttachThreadInput($currentThread, $foregroundThread, $false)
        }
    }
}

if (-not (Set-TargetWindowForeground)) {
    throw "Could not activate official simulator PID $ProcessId; no connection-panel input was sent."
}

$inputX = 0.10
$menuX = 0.04
$menuY = 0.04
$ipY = 0.168
$portY = 0.283
$buttonY = 0.377
$buttonCenterX = 0.04

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

function Test-ConnectionPanelOpen {
    # The form's IP field is a large, bright rectangle at this location when
    # the official panel is open. This remains stable across window sizes and
    # avoids relying on the small connection icon's anti-aliased pixels.
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($origin.X, $origin.Y, 0, 0, [System.Drawing.Size]::new($width, $height))
        $centerX = [int][Math]::Round($width * $inputX)
        $centerY = [int][Math]::Round($height * $ipY)
        $bright = 0
        for ($dy = -5; $dy -le 5; $dy++) {
            for ($dx = -12; $dx -le 12; $dx++) {
                $pixel = $bitmap.GetPixel($centerX + $dx, $centerY + $dy)
                if ($pixel.R -ge 220 -and $pixel.G -ge 220 -and $pixel.B -ge 220) { $bright++ }
            }
        }
        return $bright -ge 200
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

if (-not (Set-TargetWindowForeground)) {
    throw "Could not activate official simulator PID $ProcessId before inspecting its connection panel."
}

$panelOpen = Test-ConnectionPanelOpen
if (-not $panelOpen) {
    # The simulator starts with its panels collapsed. Open the left Menu Panel,
    # then re-check the button before sending any form input or connection click.
    Invoke-CanvasClick $menuX $menuY
    $panelDeadline = (Get-Date).AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 500
        $panelOpen = Test-ConnectionPanelOpen
    } while (-not $panelOpen -and (Get-Date) -lt $panelDeadline)
}
if (-not $panelOpen) {
    throw "The simulator connection panel did not open. No connection click was sent."
}

[void][AutoDriveWindowInput]::SetForegroundWindow($window)
Start-Sleep -Milliseconds 250

Invoke-CanvasClick $inputX $ipY
[System.Windows.Forms.SendKeys]::SendWait("^a")
[System.Windows.Forms.SendKeys]::SendWait($IpAddress)
Invoke-CanvasClick $inputX $portY
[System.Windows.Forms.SendKeys]::SendWait("^a")
[System.Windows.Forms.SendKeys]::SendWait([string]$Port)
Invoke-CanvasClick $buttonCenterX $buttonY

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
