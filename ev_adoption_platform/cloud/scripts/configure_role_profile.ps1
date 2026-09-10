param(
    [string]$SourceProfile = "ev-bootstrap",
    [string]$RoleProfile = "ev-deploy",
    [string]$Region = "us-east-1",
    [string]$UserName = "ev-platform-bootstrap",
    [string]$RoleName = "EVPlatformDeploymentRole"
)

$ErrorActionPreference = "Stop"

function Get-AwsCli {
    $command = Get-Command aws -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $installed = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
    if (Test-Path -LiteralPath $installed) {
        return $installed
    }

    throw "AWS CLI v2 was not found. Install it before configuring profiles."
}

$aws = Get-AwsCli
$identityJson = & $aws sts get-caller-identity --profile $SourceProfile --output json
if ($LASTEXITCODE -ne 0) {
    throw "Could not authenticate with source profile '$SourceProfile'. Run: aws configure --profile $SourceProfile"
}

$identity = $identityJson | ConvertFrom-Json
$accountId = [string]$identity.Account
$roleArn = "arn:aws:iam::${accountId}:role/${RoleName}"
$mfaArn = "arn:aws:iam::${accountId}:mfa/${UserName}"

& $aws configure set role_arn $roleArn --profile $RoleProfile
& $aws configure set source_profile $SourceProfile --profile $RoleProfile
& $aws configure set mfa_serial $mfaArn --profile $RoleProfile
& $aws configure set region $Region --profile $RoleProfile
& $aws configure set output json --profile $RoleProfile

Write-Host "Configured '$RoleProfile' to assume $roleArn with MFA."
Write-Host "Verify it with: aws sts get-caller-identity --profile $RoleProfile"
