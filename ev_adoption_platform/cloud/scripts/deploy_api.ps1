param(
    [string]$Profile = "ev-deploy",
    [string]$Region = "us-east-1",
    [string]$StackName = "EvBuyerIntelligenceDataPipeline",
    [string]$FrontendOrigins = "https://atharva2102.github.io,http://localhost:3000,http://127.0.0.1:3000"
)

$ErrorActionPreference = "Stop"
$PlatformRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IacRoot = Join-Path $PlatformRoot "cloud\iac"

function Get-AwsCli {
    $command = Get-Command aws -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $installed = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
    if (Test-Path -LiteralPath $installed) { return $installed }
    throw "AWS CLI v2 was not found."
}

function Invoke-Native {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

function Invoke-AwsJson {
    param([string[]]$Arguments)
    $raw = & $script:Aws @Arguments --profile $Profile --region $Region --output json
    if ($LASTEXITCODE -ne 0) {
        throw "AWS command failed: aws $($Arguments -join ' ')"
    }
    return $raw | ConvertFrom-Json
}

function Get-StackOutput {
    param([object]$Stack, [string]$Key)
    $entry = $Stack.Outputs | Where-Object OutputKey -eq $Key | Select-Object -First 1
    if (-not $entry) { throw "Stack output '$Key' was not found." }
    return [string]$entry.OutputValue
}

function Deploy-Stack {
    param([bool]$EnableApi, [string]$ImageTag = "latest")
    Push-Location $IacRoot
    try {
        Invoke-Native "npx" @(
            "cdk", "deploy", $StackName,
            "--require-approval", "never",
            "-c", "scheduleEnabled=false",
            "-c", "enableSnowflake=false",
            "-c", "deployApiService=$($EnableApi.ToString().ToLowerInvariant())",
            "-c", "apiImageTag=$ImageTag",
            "-c", "frontendOrigins=$FrontendOrigins"
        )
    }
    finally {
        Pop-Location
    }
}

$script:Aws = Get-AwsCli
$identity = Invoke-AwsJson @("sts", "get-caller-identity")
$accountId = [string]$identity.Account
Write-Host "Authenticated as $($identity.Arn) in account $accountId ($Region)."

$exportedJson = & $script:Aws configure export-credentials --profile $Profile --format process
if ($LASTEXITCODE -ne 0) {
    throw "Could not export temporary credentials from profile '$Profile'."
}
$exported = $exportedJson | ConvertFrom-Json
$env:AWS_ACCESS_KEY_ID = [string]$exported.AccessKeyId
$env:AWS_SECRET_ACCESS_KEY = [string]$exported.SecretAccessKey
$env:AWS_SESSION_TOKEN = [string]$exported.SessionToken
$env:AWS_DEFAULT_REGION = $Region
$env:AWS_REGION = $Region

try {
    Push-Location $IacRoot
    try {
        Invoke-Native "npm" @("ci")
        Invoke-Native "npm" @("run", "build")
        Invoke-Native "npm" @(
            "run", "synth", "--",
            "-c", "scheduleEnabled=false",
            "-c", "enableSnowflake=false",
            "-c", "deployApiService=false",
            "-c", "frontendOrigins=$FrontendOrigins"
        )
    }
    finally {
        Pop-Location
    }

    $stack = (Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)).Stacks[0]
    $repositoryOutput = $stack.Outputs | Where-Object OutputKey -eq "ApiRepositoryUri" | Select-Object -First 1
    if (-not $repositoryOutput) {
        Deploy-Stack $false
        $stack = (Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)).Stacks[0]
    }
    $repositoryUri = Get-StackOutput $stack "ApiRepositoryUri"
    $imageTag = Get-Date -Format "yyyyMMddHHmmss"

    Push-Location $PlatformRoot
    try {
        Invoke-Native "docker" @(
            "build", "--provenance=false", "-f", "api\Dockerfile",
            "-t", "ev-buyer-api:latest", "."
        )
        $password = & $script:Aws ecr get-login-password --profile $Profile --region $Region
        if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the ECR login password." }
        $registry = $repositoryUri.Split("/")[0]
        $password | & docker login --username AWS --password-stdin $registry
        if ($LASTEXITCODE -ne 0) { throw "Docker could not authenticate to ECR." }
        Invoke-Native "docker" @("tag", "ev-buyer-api:latest", "${repositoryUri}:latest")
        Invoke-Native "docker" @("tag", "ev-buyer-api:latest", "${repositoryUri}:${imageTag}")
        Invoke-Native "docker" @("push", "${repositoryUri}:latest")
        Invoke-Native "docker" @("push", "${repositoryUri}:${imageTag}")
    }
    finally {
        Pop-Location
    }

    Deploy-Stack $true $imageTag
    $stack = (Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)).Stacks[0]
    $apiUrl = Get-StackOutput $stack "ApiServiceUrl"

    Write-Host "Waiting for $apiUrl/health ..."
    $health = $null
    for ($attempt = 1; $attempt -le 24; $attempt++) {
        try {
            $health = Invoke-RestMethod "$apiUrl/health" -TimeoutSec 15
            break
        }
        catch {
            if ($attempt -eq 24) { throw }
            Start-Sleep -Seconds 10
        }
    }
    if ($health.status -ne "ok" -or $health.model_kind -ne "lightgbm") {
        throw "API health verification failed: $($health | ConvertTo-Json -Compress)"
    }

    Write-Host "API deployment succeeded: $apiUrl"
    Write-Host "Set GitHub repository variable NEXT_PUBLIC_API_BASE to $apiUrl"
    Write-Host "Set NEXT_PUBLIC_SITE_URL after GitHub Pages provides its public URL."
}
finally {
    Remove-Item Env:AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_SESSION_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_DEFAULT_REGION -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_REGION -ErrorAction SilentlyContinue
}
