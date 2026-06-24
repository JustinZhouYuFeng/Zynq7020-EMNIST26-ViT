param(
    [string]$BoardIp = "192.168.1.10",
    [string]$LocalIp = "192.168.1.100",
    [int]$UdpPort = 5001,
    [string]$SerialPort = "COM9",
    [int]$Baud = 115200,
    [string]$DataDir = "test_data\fashion_mnist_udp",
    [string]$VitisHeader = "vitis_ws\vit_qvk_test\src\tinyvit_samples_vitis.h",
    [int]$Count = 20,
    [int]$DelayMs = 1200,
    [string]$OutFile = "test_data\verify_udp_classification_batch_ps.log"
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not [System.IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $repo $DataDir
}
if (-not [System.IO.Path]::IsPathRooted($VitisHeader)) {
    $VitisHeader = Join-Path $repo $VitisHeader
}
if (-not [System.IO.Path]::IsPathRooted($OutFile)) {
    $OutFile = Join-Path $repo $OutFile
}

$manifest = Join-Path $DataDir 'manifest.csv'
if (-not (Test-Path $manifest)) {
    throw "Missing manifest: $manifest"
}

$rows = Import-Csv $manifest | Select-Object -First $Count
$expectedPreds = @()
$modelClasses = $null
$modelDataset = $null
if (Test-Path $VitisHeader) {
    $headerText = Get-Content $VitisHeader -Raw
    if ($headerText -match '#define\s+TINYVIT_NUM_CLASSES\s+([0-9]+)') {
        $modelClasses = [int]$Matches[1]
    }
    if ($headerText -match '#define\s+TINYVIT_DATASET_NAME\s+"([^"]+)"') {
        $modelDataset = $Matches[1]
    }
    $match = [regex]::Match(
        $headerText,
        'TINYVIT_SAMPLE_PYTORCH_PREDS\[[0-9]+\]\s*=\s*\{(?<body>.*?)\};',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    if ($match.Success) {
        $expectedPreds = [regex]::Matches($match.Groups['body'].Value, '-?\d+') |
            ForEach-Object { [int]$_.Value }
    }
}

$dataLeaf = Split-Path -Leaf $DataDir
$expectedClasses = $null
if ($dataLeaf -match 'emnist') {
    $expectedClasses = 26
} elseif ($dataLeaf -match 'fashion|mnist') {
    $expectedClasses = 10
}
if ($modelClasses -ne $null -and $expectedClasses -ne $null -and $modelClasses -ne $expectedClasses) {
    throw "Model/data mismatch: header classes=$modelClasses dataset=$modelDataset but DataDir=$dataLeaf expects $expectedClasses classes. Use the matching UDP dataset or regenerate tinyvit_samples_vitis.h."
}
if ($modelDataset -and $dataLeaf -match 'emnist' -and $modelDataset -notmatch 'emnist') {
    throw "Model/data mismatch: header dataset=$modelDataset but DataDir=$dataLeaf"
}
if ($modelDataset -and $dataLeaf -match 'fashion' -and $modelDataset -notmatch 'fashion') {
    throw "Model/data mismatch: header dataset=$modelDataset but DataDir=$dataLeaf"
}
Write-Host "VERIFY_MODEL dataset=$modelDataset classes=$modelClasses"
Write-Host "VERIFY_DATA dir=$DataDir rows=$($rows.Count)"
$outDir = Split-Path -Parent $OutFile
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force $outDir | Out-Null
}

$serial = [System.IO.Ports.SerialPort]::new(
    $SerialPort,
    $Baud,
    [System.IO.Ports.Parity]::None,
    8,
    [System.IO.Ports.StopBits]::One
)
$serial.ReadTimeout = 100
$serial.Open()
$serial.DiscardInBuffer()
$serial.DiscardOutBuffer()

if ($LocalIp -ne "") {
    $localEndPoint = [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Parse($LocalIp), 0)
    $client = [System.Net.Sockets.UdpClient]::new($localEndPoint)
} else {
    $client = [System.Net.Sockets.UdpClient]::new()
}
$client.Client.ReceiveTimeout = 1500

$log = New-Object System.Collections.Generic.List[string]
$summary = New-Object System.Collections.Generic.List[object]
$log.Add("[$SerialPort] OPEN`r`n")

function Read-SerialFor([int]$Milliseconds) {
    $deadline = (Get-Date).AddMilliseconds($Milliseconds)
    $textOut = New-Object System.Text.StringBuilder
    while ((Get-Date) -lt $deadline) {
        $text = $serial.ReadExisting()
        if ($text.Length -gt 0) {
            [void]$textOut.Append($text)
            $log.Add("[$SerialPort] $text")
            Write-Host -NoNewline "[$SerialPort] $text"
        }
        Start-Sleep -Milliseconds 50
    }
    return $textOut.ToString()
}

function Get-RowIndex($row) {
    if ($row.PSObject.Properties.Name -contains 'sample') {
        return [int]$row.sample
    }
    return [int]$row.index
}

try {
    [void](Read-SerialFor 800)

    foreach ($row in $rows) {
        $rowIndex = Get-RowIndex $row
        $rawPath = Join-Path $DataDir $row.raw
        $payload = [System.IO.File]::ReadAllBytes((Resolve-Path $rawPath))
        if ($payload.Length -ne 784) {
            throw "Payload must be 784 bytes, got $($payload.Length): $rawPath"
        }

        [void]$client.Send($payload, $payload.Length, $BoardIp, $UdpPort)
        $ackOk = $false
        try {
            $remote = [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Any, 0)
            $ack = $client.Receive([ref]$remote)
            $ackOk = ([System.Text.Encoding]::ASCII.GetString($ack) -eq 'ACK')
        } catch {
            $ackOk = $false
        }

        Write-Host ("`n[SEND] index={0} label={1} ack={2}" -f $rowIndex, $row.label, $ackOk)
        $text = Read-SerialFor $DelayMs
        if ($text -notmatch 'UDP_RESULT pred=0x([0-9A-Fa-f]+)') {
            $text += Read-SerialFor 4000
        }

        $pred = $null
        if ($text -match 'UDP_RESULT pred=0x([0-9A-Fa-f]+)') {
            $pred = [Convert]::ToInt32($Matches[1], 16)
        }
        $basePred = $null
        if ($text -match 'base_pred=0x([0-9A-Fa-f]+)') {
            $basePred = [Convert]::ToInt32($Matches[1], 16)
        }
        $inferMs = $null
        if ($text -match 'infer_ms=([0-9]+)') {
            $inferMs = [int]$Matches[1]
        }

        $label = [int]$row.label
        $modelPred = $null
        $modelMatch = $null
        if ($expectedPreds.Count -gt $rowIndex) {
            $modelPred = $expectedPreds[$rowIndex]
            $comparisonPred = if ($basePred -ne $null) { $basePred } else { $pred }
            $modelMatch = ($comparisonPred -eq $modelPred)
        }

        $summary.Add([pscustomobject]@{
            index = $rowIndex
            label = $label
            pred = $pred
            base_pred = $basePred
            model_pred = $modelPred
            ack = $ackOk
            correct = ($pred -eq $label)
            model_match = $modelMatch
            infer_ms = $inferMs
            raw = $row.raw
        })
    }
}
finally {
    $client.Close()
    if ($serial.IsOpen) {
        $serial.Close()
    }
    Set-Content -Path $OutFile -Value ([string]::Join('', $log)) -Encoding UTF8
}

$correct = @($summary | Where-Object { $_.correct }).Count
$modelMatched = @($summary | Where-Object { $_.model_match -eq $true }).Count
$modelComparable = @($summary | Where-Object { $_.model_match -ne $null }).Count
$timed = @($summary | Where-Object { $_.infer_ms -ne $null })
Write-Host "`nSUMMARY"
$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("`nSUMMARY")
foreach ($item in $summary) {
    $line = "index={0} label={1} pred={2} base_pred={3} model_pred={4} ack={5} correct={6} model_match={7} infer_ms={8}" -f $item.index, $item.label, $item.pred, $item.base_pred, $item.model_pred, $item.ack, $item.correct, $item.model_match, $item.infer_ms
    Write-Host $line
    $summaryLines.Add($line)
}
$line = "label_match=$correct/$($summary.Count)"
Write-Host $line
$summaryLines.Add($line)
if ($modelComparable -gt 0) {
    $line = "model_match=$modelMatched/$modelComparable"
    Write-Host $line
    $summaryLines.Add($line)
}
if ($timed.Count -gt 0) {
    $avgInfer = [math]::Round((($timed | Measure-Object -Property infer_ms -Average).Average), 2)
    $line = "infer_ms_avg=$avgInfer count=$($timed.Count)"
    Write-Host $line
    $summaryLines.Add($line)
}
Add-Content -Path $OutFile -Value $summaryLines -Encoding UTF8

if ($modelComparable -gt 0 -and $modelMatched -eq $modelComparable) {
    exit 0
}
if ($modelComparable -eq 0 -and $correct -eq $summary.Count) {
    exit 0
}
exit 1
