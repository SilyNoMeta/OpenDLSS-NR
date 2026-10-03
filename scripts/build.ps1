# Build shaders + dlss5vk.exe with the portable toolchain under tools/ and MSVC.
param([switch]$Debug)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "build_shaders.ps1")
if ($LASTEXITCODE -ne 0) { throw "shader compilation failed" }

$vcvars = & (Join-Path $PSScriptRoot "find_vcvars.ps1")
$sources = (@(Get-ChildItem (Join-Path $root "src\*.cpp")) + @(Get-ChildItem (Join-Path $root "src\d3d\*.cpp")) + @(Get-ChildItem (Join-Path $root "src\bridge\*.cpp")) | ForEach-Object { '"' + $_.FullName + '"' }) -join " "
$volk = '"' + (Join-Path $root "tools\volk\volk.c") + '"'
$include = '/I"' + (Join-Path $root "tools\Vulkan-Headers\include") + '" /I"' + (Join-Path $root "tools\volk") + '" /I"' + (Join-Path $root "src") + '"'
$opt = if ($Debug) { "/Od /Zi" } else { "/O2" }
$out = Join-Path $root "build"
New-Item -ItemType Directory -Force (Join-Path $out "obj") | Out-Null
$cmd = "`"$vcvars`" >nul && cl /nologo /std:c++20 /EHsc /W3 $opt /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS /DVK_ENABLE_BETA_EXTENSIONS /DVK_USE_PLATFORM_WIN32_KHR $include /Fo`"$out\obj\\`" $sources $volk /Fe:`"$out\dlss5vk.exe`" /link /SUBSYSTEM:CONSOLE d3d12.lib d3d11.lib dxgi.lib d3dcompiler.lib dxguid.lib"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "compilation failed" }
Write-Host "built $out\dlss5vk.exe"
