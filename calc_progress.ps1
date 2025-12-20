$ErrorActionPreference = 'Stop'
$src = 'D:\fmdd_extracted\fmdd'
$dst = 'D:\BSDS500'
$srcMeasure = Get-ChildItem -Path $src -Recurse -File | Measure-Object -Property Length -Sum
$dstMeasure = Get-ChildItem -Path $dst -Recurse -File | Measure-Object -Property Length -Sum
if ($srcMeasure.Count -eq 0) { throw 'Source dataset appears to be empty.' }
$percentCount = [math]::Round(($dstMeasure.Count / $srcMeasure.Count) * 100, 2)
$percentSize = if ($srcMeasure.Sum -eq 0) { 0 } else { [math]::Round(($dstMeasure.Sum / $srcMeasure.Sum) * 100, 2) }
[PSCustomObject]@{
    SrcFiles      = $srcMeasure.Count
    DstFiles      = $dstMeasure.Count
    SrcSizeGB     = [math]::Round($srcMeasure.Sum / 1GB, 3)
    DstSizeGB     = [math]::Round($dstMeasure.Sum / 1GB, 3)
    PercentByCount = $percentCount
    PercentBySize  = $percentSize
} | Format-List
