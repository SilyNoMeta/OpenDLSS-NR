# Runtime checks on an Ampere device; run GPU tests serially. No model/Filament dependency.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "build\tests"
$cmake = Join-Path $root "tools\cmake\bin\cmake.exe"
$ninja = Join-Path $root "tools\ninja\ninja.exe"
$vcvars = & (Join-Path $PSScriptRoot "find_vcvars.ps1")
New-Item -ItemType Directory -Force $out | Out-Null
cmd /c "`"$vcvars`" >nul && `"$cmake`" -S `"$root\tests`" -B `"$out`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release"
if ($LASTEXITCODE) { throw "test configuration failed" }
cmd /c "`"$vcvars`" >nul && `"$cmake`" --build `"$out`" --parallel 4"
if ($LASTEXITCODE) { throw "test build failed" }
& (Join-Path $out "backend-isolation.exe") (Join-Path $root "build\shaders")
if ($LASTEXITCODE) { throw "backend runtime tests failed" }
