param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$version = '2.0.0'
$dep = Join-Path $root "build/deps/duckdb-$version"
$archive = Join-Path $dep 'duckdb-shared-libs-windows-amd64.tar.gz'
$expected = '2B83169128239DE5F45FE21DE80F99BF436E53AA4BB24020DD951A1BA8D4B95F'
New-Item -ItemType Directory -Force -Path $dep | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    $uri = "https://github.com/duckdb/duckdb/releases/download/v$version/duckdb-shared-libs-windows-amd64.tar.gz"
    & curl.exe --fail --location --silent --show-error --output $archive $uri
    if ($LASTEXITCODE -ne 0) { throw "DuckDB $version C API archive download failed ($LASTEXITCODE)." }
}
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw "DuckDB archive SHA-256 mismatch: $actual" }
foreach ($name in @('duckdb.dll','duckdb.h')) {
    $target = Join-Path $dep $name
    if (-not (Test-Path -LiteralPath $target)) {
        & tar.exe -xf $archive -C $dep $name
        if ($LASTEXITCODE -ne 0) { throw "Could not extract $name from the pinned DuckDB archive." }
    }
}
Copy-Item -LiteralPath (Join-Path $dep 'duckdb.dll') -Destination (Join-Path $root 'build/duckdb.dll') -Force
Write-Host "DuckDB $version C API ready; archive SHA-256 $actual."
