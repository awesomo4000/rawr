# SPDX-License-Identifier: MPL-2.0
param([Parameter(Mandatory=$true)][string]$Zig)
$ErrorActionPreference = 'Stop'
if ($env:PROCESSOR_ARCHITECTURE -ne 'ARM64') { throw 'Expected native ARM64 PowerShell' }
$Zig = (Get-Command $Zig).Source
$logs = 'misc/portability-armwin-fallback'
New-Item -ItemType Directory -Force $logs, '.zig-cache' | Out-Null
& $Zig version | Out-File "$logs/version.txt" -Encoding utf8
& $Zig env | Out-File "$logs/compiler-env.txt" -Encoding utf8

function Invoke-Logged([string]$name, [string]$exe, [string[]]$arguments) {
    $quoted = ($arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '
    $p = Start-Process -FilePath $exe -ArgumentList $quoted -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$logs/$name.stdout.log" -RedirectStandardError "$logs/$name.log"
    Write-Host "$name exit=$($p.ExitCode)"
    return $p.ExitCode
}

function Expect-Error([string]$name, [string]$expected, [string[]]$arguments) {
    $code = Invoke-Logged $name $checker (@($Zig) + $arguments)
    if ($code -eq 0 -or !(Select-String -Quiet -SimpleMatch "error: $expected" "$logs/$name.log")) {
        throw "Expected $expected in $name"
    }
}

# The compiler may be x64; the checker must execute as ARM64 so its host gate
# selects ARM64 consumers. Its generated source also asserts that architecture.
$checker = Join-Path (Get-Location) '.zig-cache/check-package-arm.exe'
if ((Invoke-Logged 'checker-build' $Zig @('build-exe', 'check_package.zig', '-target', 'aarch64-windows-gnu', '-OReleaseSafe', "-femit-bin=$checker")) -ne 0) { throw 'Checker build failed' }
Expect-Error 'override' 'RunTargetOverride' @('--target', 'aarch64-windows-gnu')
Expect-Error 'msvc-override' 'RunTargetOverride' @('--run-windows-msvc', '--cpu', 'baseline')
Expect-Error 'gnu-override' 'RunTargetOverride' @('--run-windows-gnu', '--target', 'aarch64-windows-gnu')
Expect-Error 'msvc-build-only' 'MsvcRequiresExecution' @('--run-windows-msvc', '--build-only')
Expect-Error 'gnu-build-only' 'WindowsGnuRequiresExecution' @('--run-windows-gnu', '--build-only')
Expect-Error 'msvc-duplicate' 'DuplicateMsvcMode' @('--run-windows-msvc', '--run-windows-msvc')
Expect-Error 'gnu-duplicate' 'DuplicateWindowsGnuMode' @('--run-windows-gnu', '--run-windows-gnu')
Expect-Error 'mixed' 'ConflictingExecutionModes' @('--run-windows-msvc', '--run-windows-gnu')

$original = Get-Content -Raw check_package.zig
$mutation = "check_package_arm_control_$([Guid]::NewGuid().ToString('N')).zig"
$mutationExe = Join-Path (Get-Location) '.zig-cache/check-package-arm-control.exe'
try {
    $cases = @(
        @('msvc-as-gnu', 'aarch64-windows-msvc', 'aarch64-windows-gnu', '--run-windows-msvc', 'ExpectedWindowsMsvcConsumer'),
        @('gnu-as-msvc', 'aarch64-windows-gnu', 'aarch64-windows-msvc', '--run-windows-gnu', 'ExpectedWindowsGnuConsumer'),
        @('wrong-arch', 'aarch64-windows-gnu', 'x86_64-windows-gnu', '--run-windows-gnu', 'ExpectedConsumerArchitecture')
    )
    foreach ($case in $cases) {
        [IO.File]::WriteAllText((Join-Path (Get-Location) $mutation), $original.Replace('=> "' + $case[1] + '"', '=> "' + $case[2] + '"'), [Text.Encoding]::UTF8)
        if ((Invoke-Logged "$($case[0])-build" $Zig @('build-exe', $mutation, '-target', 'aarch64-windows-gnu', '-OReleaseSafe', "-femit-bin=$mutationExe")) -ne 0) { throw 'Mutation build failed' }
        $code = Invoke-Logged $case[0] $mutationExe @($Zig, $case[3], '--scratch-suffix', 'arm-control')
        if ($code -eq 0 -or !(Select-String -Quiet -SimpleMatch $case[4] "$logs/$($case[0]).log")) { throw "Mutation not detected: $($case[0])" }
    }
} finally {
    Remove-Item -LiteralPath $mutation -ErrorAction SilentlyContinue
}

$failed = $false
foreach ($abi in @('gnu', 'msvc')) {
    if ((Invoke-Logged "$abi-package" $checker @($Zig, "--run-windows-$abi")) -ne 0) { $failed = $true }
    foreach ($step in @('test', 'test64', 'difftest', 'difftest64')) {
        if ((Invoke-Logged "$abi-$step" $Zig @('build', $step, "-Dtarget=aarch64-windows-$abi", '--cache-dir', '.zig-cache-x64-driver', '--summary', 'all')) -ne 0) { $failed = $true }
    }
}
if ($failed) { exit 1 }
