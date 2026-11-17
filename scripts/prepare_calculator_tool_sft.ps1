param([int]$TrainExamples=256,[int]$ValidationExamples=64,[string]$OutputDirectory='data/processed/calculator_tool_sft')
$ErrorActionPreference='Stop'
if($TrainExamples -lt 16 -or $ValidationExamples -lt 8){throw 'Use at least 16 training and 8 validation examples.'}
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Force -Path $out | Out-Null
$rng=[Random]::new(20261117)
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
function Escape-Field([string]$value){return $value.Replace('\','\\').Replace("`t",'\t').Replace("`r",'\r').Replace("`n",'\n')}
function Make-Example {
    while($true){
        $a=$rng.Next(2,1000);$b=$rng.Next(2,1000);$kind=$rng.Next(0,4)
        switch($kind){
            0{$op='+';$answer=$a+$b}
            1{$op='-';$answer=$a-$b}
            2{$op='*';$answer=$a*$b}
            default{$op='/';$b=$rng.Next(2,100);$a=$b*$rng.Next(2,100);$answer=$a/$b}
        }
        $expression="$a$op$b"
        if($seen.Add($expression)){break}
    }
    $prompt="User: Use the calculator tool to compute $a $op $b.`nAssistant:"
    $response="<|tool_call|>calculator`t$expression<|end_tool_call|>`n<|tool_result|>calculator`n[untrusted tool output]`n$answer<|end_tool_result|>`nThe answer is $answer."
    return (Escape-Field $prompt)+"`t"+(Escape-Field $response)
}
$train=for($i=0;$i -lt $TrainExamples;$i++){Make-Example}
$valid=for($i=0;$i -lt $ValidationExamples;$i++){Make-Example}
$encoding=[Text.UTF8Encoding]::new($false)
$trainPath=Join-Path $out 'train.tsv';$validationPath=Join-Path $out 'validation.tsv'
[IO.File]::WriteAllLines($trainPath,$train,$encoding);[IO.File]::WriteAllLines($validationPath,$valid,$encoding)
$metadata=[ordered]@{format='ama10-calculator-tool-sft-v1';kind='synthetic deterministic arithmetic tool traces';seed=20261117;train_examples=$TrainExamples;validation_examples=$ValidationExamples;tool_protocol='<|tool_call|>calculator<TAB>expression<|end_tool_call|>';tool_result='calculator output framed as untrusted tool output';train_sha256=(Get-FileHash -LiteralPath $trainPath -Algorithm SHA256).Hash;validation_sha256=(Get-FileHash -LiteralPath $validationPath -Algorithm SHA256).Hash}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'sources.json') -Encoding utf8
Write-Output "prepared_train=$TrainExamples validation=$ValidationExamples output=$out"
