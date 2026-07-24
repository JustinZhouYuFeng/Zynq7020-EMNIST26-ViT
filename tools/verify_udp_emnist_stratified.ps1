param(
    [string]$BoardIp = "192.168.1.10",
    [string]$LocalIp = "192.168.1.100",
    [int]$UdpPort = 5001,
    [string]$SerialPort = "COM9",
    [int]$Baud = 115200,
    [string]$DataDir = "test_data\emnist_letters_stratified_520",
    [int]$Count = 0,
    [int]$ResultTimeoutMs = 2000,
    [int]$InterSampleDelayMs = 50,
    [string]$OutDir = "validation_results\board"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
foreach ($name in @('DataDir', 'OutDir')) {
    $value = Get-Variable -Name $name -ValueOnly
    if (-not [IO.Path]::IsPathRooted($value)) {
        Set-Variable -Name $name -Value (Join-Path $repo $value)
    }
}
$manifest = Join-Path $DataDir 'manifest.csv'
if (-not (Test-Path $manifest)) { throw "Missing manifest: $manifest" }
$rows = @(Import-Csv $manifest)
if ($Count -gt 0) { $rows = @($rows | Select-Object -First $Count) }
if ($rows.Count -eq 0) { throw 'No validation rows selected.' }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$serial = [IO.Ports.SerialPort]::new(
    $SerialPort, $Baud, [IO.Ports.Parity]::None, 8, [IO.Ports.StopBits]::One
)
$serial.ReadTimeout = 100
$serial.Open()
$serial.DiscardInBuffer()
$client = [Net.Sockets.UdpClient]::new(
    [Net.IPEndPoint]::new([Net.IPAddress]::Parse($LocalIp), 0)
)
$client.Client.ReceiveTimeout = 1000
$results = [Collections.Generic.List[object]]::new()
$rawLog = [Text.StringBuilder]::new()

function Wait-BoardResult([int]$TimeoutMs) {
    $deadline = (Get-Date).AddMilliseconds($TimeoutMs)
    $buffer = [Text.StringBuilder]::new()
    while ((Get-Date) -lt $deadline) {
        $chunk = $serial.ReadExisting()
        if ($chunk.Length -gt 0) {
            [void]$buffer.Append($chunk)
            [void]$rawLog.Append($chunk)
            Write-Host -NoNewline $chunk
            if ($buffer.ToString() -match 'UDP_RESULT[^\r\n]*') {
                return $Matches[0]
            }
        }
        Start-Sleep -Milliseconds 10
    }
    return $null
}

try {
    Start-Sleep -Milliseconds 500
    $serial.DiscardInBuffer()
    foreach ($row in $rows) {
        $payload = [IO.File]::ReadAllBytes((Join-Path $DataDir $row.raw))
        if ($payload.Length -ne 784) { throw "Bad payload length: $($row.raw)" }
        [void]$client.Send($payload, $payload.Length, $BoardIp, $UdpPort)
        $ack = $false
        try {
            $remote = [Net.IPEndPoint]::new([Net.IPAddress]::Any, 0)
            $reply = $client.Receive([ref]$remote)
            $ack = ([Text.Encoding]::ASCII.GetString($reply) -eq 'ACK')
        } catch { $ack = $false }

        $line = Wait-BoardResult $ResultTimeoutMs
        $pred = $null; $basePred = $null
        $inferMs = $null; $patchMs = $null; $stackMs = $null; $headMs = $null
        if ($line -match 'pred=0x([0-9A-Fa-f]+)') { $pred = [Convert]::ToInt32($Matches[1], 16) }
        if ($line -match 'base_pred=0x([0-9A-Fa-f]+)') { $basePred = [Convert]::ToInt32($Matches[1], 16) }
        if ($line -match 'infer_ms=([0-9]+)') { $inferMs = [int]$Matches[1] }
        if ($line -match 'patch_ms=([0-9]+)') { $patchMs = [int]$Matches[1] }
        if ($line -match 'pl_stack_ms=([0-9]+)') { $stackMs = [int]$Matches[1] }
        if ($line -match 'head_ms=([0-9]+)') { $headMs = [int]$Matches[1] }
        $label = [int]$row.label
        $golden = [int]$row.pytorch_pred
        $results.Add([pscustomobject]@{
            sample = [int]$row.sample; source_index = [int]$row.source_index
            label = $label; pytorch_pred = $golden; board_pred = $pred
            board_base_pred = $basePred; ack = $ack
            label_correct = ($pred -eq $label)
            base_label_correct = ($basePred -eq $label)
            hardware_software_match = ($basePred -eq $golden)
            infer_ms = $inferMs; patch_ms = $patchMs; pl_stack_ms = $stackMs
            head_ms = $headMs; raw = $row.raw
        })
        Write-Host ("`n[{0}/{1}] label={2} pred={3} base={4} golden={5} ack={6}" -f
            $results.Count, $rows.Count, $label, $pred, $basePred, $golden, $ack)
        Start-Sleep -Milliseconds $InterSampleDelayMs
    }

    $drainDeadline = (Get-Date).AddMilliseconds(500)
    while ((Get-Date) -lt $drainDeadline) {
        $tail = $serial.ReadExisting()
        if ($tail.Length -gt 0) {
            [void]$rawLog.Append($tail)
            Write-Host -NoNewline $tail
        }
        $completeCount = [regex]::Matches(
            $rawLog.ToString(), 'UDP_RESULT[^\r\n]*head_ms=[0-9]+'
        ).Count
        if ($completeCount -ge $rows.Count) { break }
        Start-Sleep -Milliseconds 10
    }
}
finally {
    $client.Close()
    if ($serial.IsOpen) { $serial.Close() }
    [IO.File]::WriteAllText((Join-Path $OutDir 'uart.log'), $rawLog.ToString())
}

$completeLines = [regex]::Matches(
    $rawLog.ToString(), 'UDP_RESULT[^\r\n]*head_ms=[0-9]+'
)
if ($completeLines.Count -eq $results.Count) {
    for ($index = 0; $index -lt $results.Count; $index++) {
        $line = $completeLines[$index].Value
        $predMatch = [regex]::Match($line, '(?:^|\s)pred=0x([0-9A-Fa-f]+)')
        $baseMatch = [regex]::Match($line, 'base_pred=0x([0-9A-Fa-f]+)')
        $inferMatch = [regex]::Match($line, 'infer_ms=([0-9]+)')
        $patchMatch = [regex]::Match($line, 'patch_ms=([0-9]+)')
        $stackMatch = [regex]::Match($line, 'pl_stack_ms=([0-9]+)')
        $headMatch = [regex]::Match($line, 'head_ms=([0-9]+)')
        if ($predMatch.Success) {
            $results[$index].board_pred = [Convert]::ToInt32($predMatch.Groups[1].Value, 16)
        }
        if ($baseMatch.Success) {
            $results[$index].board_base_pred = [Convert]::ToInt32($baseMatch.Groups[1].Value, 16)
        }
        if ($inferMatch.Success) { $results[$index].infer_ms = [int]$inferMatch.Groups[1].Value }
        if ($patchMatch.Success) { $results[$index].patch_ms = [int]$patchMatch.Groups[1].Value }
        if ($stackMatch.Success) { $results[$index].pl_stack_ms = [int]$stackMatch.Groups[1].Value }
        if ($headMatch.Success) { $results[$index].head_ms = [int]$headMatch.Groups[1].Value }
        $results[$index].label_correct = (
            $results[$index].board_pred -eq $results[$index].label
        )
        $results[$index].base_label_correct = (
            $results[$index].board_base_pred -eq $results[$index].label
        )
        $results[$index].hardware_software_match = (
            $results[$index].board_base_pred -eq $results[$index].pytorch_pred
        )
    }
} else {
    Write-Warning (
        "Complete UART result count {0} does not match sample count {1}." -f
        $completeLines.Count, $results.Count
    )
}

$results | Export-Csv (Join-Path $OutDir 'predictions.csv') -NoTypeInformation -Encoding UTF8
$valid = @($results | Where-Object { $_.board_pred -ne $null })
$timed = @($results | Where-Object { $_.infer_ms -ne $null } | Sort-Object infer_ms)
if ($valid.Count -eq 0) {
    @(
        "samples=$($results.Count)"
        "result_received=0/$($results.Count)"
        "status=FAILED_NO_BOARD_RESULTS"
        "note=No UDP_RESULT lines were received; accuracy is not measurable."
    ) | Set-Content (Join-Path $OutDir 'summary.txt') -Encoding UTF8
    throw "No board results received. Check power, Ethernet link, firmware, and UART."
}
function Rate($items, [string]$property) {
    if ($items.Count -eq 0) { return 0.0 }
    return 100.0 * @($items | Where-Object { $_.$property -eq $true }).Count / $items.Count
}
function Percentile($items, [string]$property, [double]$p) {
    if ($items.Count -eq 0) { return $null }
    $index = [Math]::Min($items.Count - 1, [Math]::Floor(($items.Count - 1) * $p))
    return $items[$index].$property
}

$summary = [Collections.Generic.List[string]]::new()
$summary.Add("samples=$($results.Count)")
$summary.Add(("result_received={0}/{1}" -f $valid.Count, $results.Count))
$summary.Add(("ack_rate={0:N2}%" -f (Rate $results 'ack')))
$summary.Add(("board_final_accuracy={0:N2}%" -f (Rate $valid 'label_correct')))
$summary.Add(("board_base_accuracy={0:N2}%" -f (Rate $valid 'base_label_correct')))
$summary.Add(("hardware_software_agreement={0:N2}%" -f (Rate $valid 'hardware_software_match')))
if ($timed.Count -gt 0) {
    $summary.Add(("infer_ms_avg={0:N2}" -f (($timed | Measure-Object infer_ms -Average).Average)))
    $summary.Add("infer_ms_p50=$(Percentile $timed 'infer_ms' 0.50)")
    $summary.Add("infer_ms_p95=$(Percentile $timed 'infer_ms' 0.95)")
}
for ($label = 0; $label -lt 26; $label++) {
    $classRows = @($valid | Where-Object { $_.label -eq $label })
    $summary.Add(("class_{0}_{1}={2:N2}% ({3})" -f
        $label, [char](65 + $label), (Rate $classRows 'label_correct'), $classRows.Count))
}
$summary | Set-Content (Join-Path $OutDir 'summary.txt') -Encoding UTF8
$summary | ForEach-Object { Write-Host $_ }
