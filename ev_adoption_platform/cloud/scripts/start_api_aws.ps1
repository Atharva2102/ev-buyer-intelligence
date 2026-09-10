param(
    [string]$Profile = "ev-deploy",
    [string]$Region = "us-east-1",
    [string]$StackName = "EvBuyerIntelligenceDataPipeline",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$awsCommand = Get-Command aws -ErrorAction SilentlyContinue
$aws = if ($awsCommand) { $awsCommand.Source } else { "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }
if (-not (Test-Path -LiteralPath $aws)) { throw "AWS CLI v2 was not found." }

$credentials = (& $aws configure export-credentials --profile $Profile --format process) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Could not export credentials from '$Profile'." }
$env:AWS_ACCESS_KEY_ID = [string]$credentials.AccessKeyId
$env:AWS_SECRET_ACCESS_KEY = [string]$credentials.SecretAccessKey
$env:AWS_SESSION_TOKEN = [string]$credentials.SessionToken
$env:AWS_DEFAULT_REGION = $Region
$env:SCORING_EVENTS_TABLE = & $aws cloudformation describe-stacks `
    --stack-name $StackName `
    --profile $Profile `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='ScoringEventsTableName'].OutputValue | [0]" `
    --output text

if (-not $env:SCORING_EVENTS_TABLE -or $env:SCORING_EVENTS_TABLE -eq "None") {
    throw "The stack does not expose a scoring events table. Deploy the latest CDK stack first."
}

Write-Host "Starting API with DynamoDB event storage: $env:SCORING_EVENTS_TABLE"
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port $Port
