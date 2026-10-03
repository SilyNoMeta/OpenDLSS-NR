# Build and run the Direct3D CUDA launch probe (tests/d3d/cuda_probe.cpp) on the NVIDIA adapter, for D3D12 and
# D3D11. Needs MSVC and an NVIDIA driver; ptxas (CUDA Toolkit) is optional and only adds a cubin for D3D11 in case
# the driver refuses PTX text there. Run GPU tests serially.
#   powershell -File scripts\test_d3d_probe.ps1 [-Debug] [-Out build\d3d]
param([switch]$Debug, [string]$Out = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root "build\d3d" }
New-Item -ItemType Directory -Force (Join-Path $Out "obj") | Out-Null
$vcvars = & (Join-Path $PSScriptRoot "find_vcvars.ps1")
$source = Join-Path $root "tests\d3d\cuda_probe.cpp"
$exe = Join-Path $Out "cuda-probe.exe"
$cmd = "`"$vcvars`" >nul && cl /nologo /std:c++20 /EHsc /W3 /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /Fo`"$Out\obj\\`" `"$source`" /Fe:`"$exe`" /link /SUBSYSTEM:CONSOLE d3d12.lib d3d11.lib dxgi.lib d3dcompiler.lib dxguid.lib"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "cuda-probe compilation failed" }

$ptx = Join-Path $root "tests\d3d\probe_kernels.ptx"
$cubin = Join-Path $Out "probe_kernels.sm_80.cubin"
$ptxas = Get-Command ptxas -ErrorAction SilentlyContinue
if ($ptxas) {
  & $ptxas.Source -arch=sm_80 -O3 $ptx -o $cubin
  if ($LASTEXITCODE -ne 0) { throw "ptxas failed" }
}
$failed = $false
foreach ($api in "d3d12", "d3d11") {
  $arguments = @("--api", $api, "--ptx", $ptx, "--json", (Join-Path $Out "probe-$api.json"))
  if ($api -eq "d3d11" -and (Test-Path $cubin)) { $arguments += @("--cubin", $cubin) }
  if ($Debug) { $arguments += "--debug" }
  & $exe @arguments
  if ($LASTEXITCODE -ne 0) { $failed = $true }
}
if ($failed) { throw "Direct3D CUDA probe failed" }
