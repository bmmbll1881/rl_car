# Validate that a media file can be opened/decoded by the Windows media stack (WinRT).
param([Parameter(Mandatory=$true)][string]$Path)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Editing.MediaClip, Windows.Media.Editing, ContentType = WindowsRuntime]
$null = [Windows.Media.Editing.MediaComposition, Windows.Media.Editing, ContentType = WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await($op, [Type]$inner) {
    $task = $asTaskGeneric.MakeGenericMethod($inner).Invoke($null, @($op))
    try {
        $task.Wait(-1) | Out-Null
    } catch {
        Write-Host "AWAIT FAILED: $($_.Exception.InnerException.Message)"
        Write-Host "  inner: $($_.Exception.InnerException.InnerException.Message)"
        return $null
    }
    return $task.Result
}

$full = (Resolve-Path $Path).Path
Write-Host "opening: $full"
$getFile = [Windows.Storage.StorageFile].GetMethod('GetFileFromPathAsync', [Type[]]@([string]))
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($full)) $getFile.ReturnType.GetGenericArguments()[0]
if (-not $file) { Write-Host 'RESULT: cannot open file'; exit 1 }
Write-Host "  storage file ok: $($file.Name), $($file.DisplayType)"

$createClip = [Windows.Media.Editing.MediaClip].GetMethod('CreateFromFileAsync')
$clip = Await ([Windows.Media.Editing.MediaClip]::CreateFromFileAsync($file)) $createClip.ReturnType.GetGenericArguments()[0]
if (-not $clip) { Write-Host 'RESULT: NO - media stack could not decode this file'; exit 1 }

Write-Host "  clip ok: original duration = $($clip.OriginalDuration)"
$comp = [Windows.Media.Editing.MediaComposition]::new()
$comp.Clips.Add($clip)
Write-Host "  composition ok: clips=$($comp.Clips.Count) duration=$($comp.Duration)"
Write-Host 'RESULT: OK - Windows media stack decoded the file'
