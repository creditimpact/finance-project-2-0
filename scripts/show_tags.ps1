param(
    [Parameter(Mandatory = $true)]
    [string]$sid
)

$ErrorActionPreference = "Stop"

python -m scripts.show_tags --sid $sid
