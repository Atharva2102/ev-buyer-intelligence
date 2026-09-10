param(
    [string]$Profile = "ev-deploy",
    [string]$Region = "us-east-1",
    [string]$StackName = "EvBuyerIntelligenceDataPipeline",
    [string]$FrontendOrigins = "https://atharva2102.github.io,http://localhost:3000,http://127.0.0.1:3000"
)

$ErrorActionPreference = "Stop"
$env:PAGER = ""
$env:AWS_PAGER = ""
$PlatformRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IacRoot = Join-Path $PlatformRoot "cloud\iac"

function Invoke-Native {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

function Invoke-AwsJson {
    param([string[]]$Arguments)
    $raw = & aws @Arguments --profile $Profile --region $Region --no-cli-pager --output json
    if ($LASTEXITCODE -ne 0) { throw "AWS command failed: aws $($Arguments -join ' ')" }
    return $raw | ConvertFrom-Json
}

function Get-StackOutput {
    param([object]$Stack, [string]$Key)
    $entry = $Stack.Outputs | Where-Object OutputKey -eq $Key | Select-Object -First 1
    if (-not $entry) { throw "Stack output '$Key' was not found." }
    return [string]$entry.OutputValue
}

$identity = Invoke-AwsJson @("sts", "get-caller-identity")
Write-Host "Authenticated as $($identity.Arn) in account $($identity.Account) ($Region)."

$exportedJson = & aws configure export-credentials --profile $Profile --format process
if ($LASTEXITCODE -ne 0) { throw "Could not export temporary credentials." }
$exported = $exportedJson | ConvertFrom-Json
$env:AWS_ACCESS_KEY_ID = [string]$exported.AccessKeyId
$env:AWS_SECRET_ACCESS_KEY = [string]$exported.SecretAccessKey
$env:AWS_SESSION_TOKEN = [string]$exported.SessionToken
$env:AWS_DEFAULT_REGION = $Region
$env:AWS_REGION = $Region

try {
    $stack = (Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)).Stacks[0]
    $repositoryUri = Get-StackOutput $stack "ApiRepositoryUri"
    $imageTag = "lambda-$(Get-Date -Format 'yyyyMMddHHmmss')"

    Push-Location $PlatformRoot
    try {
        Invoke-Native "docker" @(
            "build", "--platform", "linux/amd64", "--provenance=false",
            "-f", "api\Dockerfile.lambda", "-t", "ev-buyer-api-lambda:latest", "."
        )
        $password = & aws ecr get-login-password --profile $Profile --region $Region
        if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the ECR login password." }
        $registry = $repositoryUri.Split("/")[0]
        $password | & docker login --username AWS --password-stdin $registry
        if ($LASTEXITCODE -ne 0) { throw "Docker could not authenticate to ECR." }
        Invoke-Native "docker" @("tag", "ev-buyer-api-lambda:latest", "${repositoryUri}:lambda-latest")
        Invoke-Native "docker" @("tag", "ev-buyer-api-lambda:latest", "${repositoryUri}:${imageTag}")
        Invoke-Native "docker" @("push", "${repositoryUri}:lambda-latest")
        Invoke-Native "docker" @("push", "${repositoryUri}:${imageTag}")
    }
    finally { Pop-Location }

    Push-Location $IacRoot
    try {
        Invoke-Native "npm" @("run", "build")
        Invoke-Native "npx" @(
            "cdk", "deploy", $StackName, "--require-approval", "never",
            "-c", "scheduleEnabled=false", "-c", "enableSnowflake=false",
            "-c", "deployApiService=false", "-c", "deployApiLambda=true",
            "-c", "apiImageTag=$imageTag", "-c", "frontendOrigins=$FrontendOrigins"
        )
    }
    finally { Pop-Location }

    $stack = (Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)).Stacks[0]
    $apiUrl = (Get-StackOutput $stack "ApiLambdaUrl").TrimEnd("/")
    $health = $null
    for ($attempt = 1; $attempt -le 18; $attempt++) {
        try {
            $health = Invoke-RestMethod "$apiUrl/health" -TimeoutSec 100
            break
        }
        catch {
            if ($attempt -eq 18) { throw }
            Start-Sleep -Seconds 10
        }
    }
    if ($health.status -ne "ok" -or $health.model_kind -ne "lightgbm") {
        throw "API health verification failed: $($health | ConvertTo-Json -Compress)"
    }

    Write-Host "Lambda API deployment succeeded: $apiUrl"
    Write-Host "Set GitHub repository variable NEXT_PUBLIC_API_BASE to $apiUrl"
}
finally {
    Remove-Item Env:AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_SESSION_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_DEFAULT_REGION -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_REGION -ErrorAction SilentlyContinue
}
