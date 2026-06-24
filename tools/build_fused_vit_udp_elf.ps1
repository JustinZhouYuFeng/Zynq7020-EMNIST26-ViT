param(
    [string]$OutName = "vit_qvk_test_udp_fused.elf",
    [ValidateSet('auto', 'fixed')]
    [string]$PhyMode = 'auto'
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$vitis = 'D:\fpga\Vitis\2020.1'
$src = Join-Path $repo 'vitis_ws\vit_qvk_test\src'
$fusedBsp = Join-Path $repo 'vitis_ws\vit_zynq_fused_platform\ps7_cortexa9_0\standalone_domain\bsp\ps7_cortexa9_0'
$netBsp = Join-Path $repo 'vitis_ws\vit_zynq_platform\ps7_cortexa9_0\standalone_domain\bsp\ps7_cortexa9_0'
$out = Join-Path $repo 'vitis_ws\vit_qvk_test\Debug_fused_manual'

$env:PATH = "$vitis\gnu\aarch32\nt\gcc-arm-none-eabi\bin;$vitis\gnuwin\bin;" + $env:PATH

New-Item -ItemType Directory -Force $out | Out-Null

$compileFlags = @(
    '-O0', '-g3', '-Wall', '-c', '-fmessage-length=0',
    '-MMD', '-MP',
    '-mcpu=cortex-a9', '-mfpu=vfpv3', '-mfloat-abi=hard',
    "-I$src",
    "-I$netBsp\include",
    "-I$fusedBsp\include"
)

$sources = @('main.c', 'platform_zynq.c')
if ($PhyMode -eq 'fixed') {
    $sources += 'xemacpsif_physpeed_fixed.c'
}

$objects = @()
foreach ($source in $sources) {
    $object = Join-Path $out ($source -replace '\.c$', '.o')
    & arm-none-eabi-gcc @compileFlags -o $object (Join-Path $src $source)
    $compileExit = $LASTEXITCODE
    if ($compileExit -ne 0) {
        throw "compile failed: $source"
    }
    $objects += $object
}

$elf = Join-Path $out $OutName
$linkArgs = @('-o', $elf) + $objects + @(
    '-Wl,-T', "-Wl,$(Join-Path $src 'lscript.ld')",
    "-L$netBsp\lib",
    "-L$fusedBsp\lib",
    '-mcpu=cortex-a9', '-mfpu=vfpv3', '-mfloat-abi=hard',
    '-Wl,-build-id=none', "-specs=$src\Xilinx.spec",
    '-Wl,--start-group', '-lxil', '-llwip4', '-lgcc', '-lc', '-Wl,--end-group'
)

& arm-none-eabi-gcc @linkArgs
$linkExit = $LASTEXITCODE
if ($linkExit -ne 0) {
    throw 'link failed'
}

& arm-none-eabi-size $elf
$sizeExit = $LASTEXITCODE
if ($sizeExit -ne 0) {
    throw 'size failed'
}
Write-Host "FUSED_ELF=$elf"
