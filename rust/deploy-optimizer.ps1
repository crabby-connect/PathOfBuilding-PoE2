# Build the pob-optimizer cdylib (release) and deploy it next to the other runtime
# DLLs, where the GUI's loader path can find it (OptimizerPool.lua does
# ffi.load("pob_optimizer"), which resolves pob_optimizer.dll off PATH; runtime/ is
# already on PATH when the GUI runs).
#
# The DLL is a build artifact — NOT checked in (see runtime/.gitignore-style entry
# in the repo .gitignore). Re-run this after changing anything under rust/pob-optimizer.
#
# Usage (from anywhere):
#   pwsh rust/deploy-optimizer.ps1            # release build + copy
#   pwsh rust/deploy-optimizer.ps1 -SkipBuild # just copy an already-built DLL
#
# Exits non-zero on build or copy failure so CI can gate on it.

[CmdletBinding()]
param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's rust/ directory.
$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repoRoot "rust/pob-optimizer/Cargo.toml"
$dll      = Join-Path $repoRoot "rust/pob-optimizer/target/release/pob_optimizer.dll"
$runtime  = Join-Path $repoRoot "runtime"

if (-not (Test-Path $manifest)) {
    throw "manifest not found: $manifest (run from the repo, with rust/pob-optimizer present)"
}
if (-not (Test-Path $runtime)) {
    throw "runtime dir not found: $runtime"
}

if (-not $SkipBuild) {
    Write-Host "Building pob-optimizer (release) ..."
    & cargo build --release --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
}

if (-not (Test-Path $dll)) {
    throw "cdylib not found: $dll (build it first, or omit -SkipBuild)"
}

$dest = Join-Path $runtime "pob_optimizer.dll"
Copy-Item -Path $dll -Destination $dest -Force
Write-Host "Deployed -> $dest"
