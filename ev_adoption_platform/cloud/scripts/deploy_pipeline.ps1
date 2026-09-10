param(
    [string]$Profile = "ev-deploy",
    [string]$Region = "us-east-1",
    [string]$StackName = "EvBuyerIntelligenceDataPipeline",
    [string]$ScoresPath = "",
    [switch]$Bootstrap,
    [switch]$SkipInfrastructure,
    [switch]$SkipImagePush,
    [switch]$SkipInputUpload
)

$ErrorActionPreference = "Stop"
$PlatformRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RepoRoot = (Resolve-Path (Join-Path $PlatformRoot "..")).Path
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

$script:Aws = Get-AwsCli
$identity = Invoke-AwsJson @("sts", "get-caller-identity")
$accountId = [string]$identity.Account
Write-Host "Authenticated as $($identity.Arn) in account $accountId ($Region)."

# CDK's Node credential provider cannot prompt for an MFA-backed role profile.
# Export the AWS CLI's cached temporary role session for CDK, then clear it.
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

if (-not $SkipInfrastructure) {
    Push-Location $IacRoot
    try {
        Invoke-Native "npm" @("ci")
        Invoke-Native "npm" @("run", "build")
        Invoke-Native "npm" @("run", "synth", "--", "-c", "scheduleEnabled=false", "-c", "enableSnowflake=false")
        if ($Bootstrap) {
            Invoke-Native "npx" @("cdk", "bootstrap", "aws://$accountId/$Region")
        }
        Invoke-Native "npx" @(
            "cdk", "deploy", $StackName,
            "--require-approval", "never",
            "-c", "scheduleEnabled=false",
            "-c", "enableSnowflake=false"
        )
    }
    finally {
        Pop-Location
    }
}

$stackResponse = Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)
$stack = $stackResponse.Stacks[0]
$bucket = Get-StackOutput $stack "DataBucketName"
$repositoryUri = Get-StackOutput $stack "PipelineRepositoryUri"
$cluster = Get-StackOutput $stack "ClusterName"
$taskDefinition = Get-StackOutput $stack "TaskDefinitionArn"
$subnets = (Get-StackOutput $stack "PublicSubnetIds") -split ","
$securityGroup = Get-StackOutput $stack "SecurityGroupId"
$scheduleEnabled = Get-StackOutput $stack "ScheduleEnabled"
if ($scheduleEnabled -ne "false") {
    throw "Safety check failed: the EventBridge schedule is enabled."
}

Push-Location $PlatformRoot
try {
    if (-not $SkipImagePush) {
        Invoke-Native "docker" @("build", "--provenance=false", "-f", "cloud\pipeline\Dockerfile", "-t", "ev-buyer-pipeline:latest", ".")
        $password = & $script:Aws ecr get-login-password --profile $Profile --region $Region
        if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the ECR login password." }
        $registry = $repositoryUri.Split("/")[0]
        $password | & docker login --username AWS --password-stdin $registry
        if ($LASTEXITCODE -ne 0) { throw "Docker could not authenticate to ECR." }
        Invoke-Native "docker" @("tag", "ev-buyer-pipeline:latest", "${repositoryUri}:latest")
        Invoke-Native "docker" @("push", "${repositoryUri}:latest")
    }

    if (-not $SkipInputUpload) {
        if (-not $ScoresPath) {
            $ScoresPath = Join-Path $RepoRoot "kaggle_competition_workspace\submissions\top_10\submission_chris_lgbm_tuned_blend.csv"
        }
        $env:AWS_PROFILE = $Profile
        $env:AWS_DEFAULT_REGION = $Region
        try {
            Invoke-Native "python" @(
                "cloud\scripts\upload_inputs.py",
                "--bucket", $bucket,
                "--train", (Join-Path $RepoRoot "kaggle_competition_workspace\data\train.csv"),
                "--test", (Join-Path $RepoRoot "kaggle_competition_workspace\data\test.csv"),
                "--original", (Join-Path $RepoRoot "kaggle_competition_workspace\data\OG Dataset\EV_Adoption_and_Range_Anxiety_Dataset.csv"),
                "--scores", (Resolve-Path $ScoresPath).Path
            )
        }
        finally {
            Remove-Item Env:AWS_PROFILE -ErrorAction SilentlyContinue
            Remove-Item Env:AWS_DEFAULT_REGION -ErrorAction SilentlyContinue
        }
    }
}
finally {
    Pop-Location
}

$networkConfiguration = "awsvpcConfiguration={subnets=[$($subnets -join ',')],securityGroups=[$securityGroup],assignPublicIp=ENABLED}"

$run = Invoke-AwsJson @(
    "ecs", "run-task",
    "--cluster", $cluster,
    "--task-definition", $taskDefinition,
    "--launch-type", "FARGATE",
    "--network-configuration", $networkConfiguration
)
if (-not $run.tasks -or $run.failures.Count -gt 0) {
    throw "ECS did not start the task: $($run.failures | ConvertTo-Json -Compress)"
}
$taskArn = [string]$run.tasks[0].taskArn
Write-Host "Started $taskArn. Waiting for completion..."
Invoke-Native $script:Aws @("ecs", "wait", "tasks-stopped", "--cluster", $cluster, "--tasks", $taskArn, "--profile", $Profile, "--region", $Region)

$task = (Invoke-AwsJson @("ecs", "describe-tasks", "--cluster", $cluster, "--tasks", $taskArn)).tasks[0]
$container = $task.containers[0]
if ([int]$container.exitCode -ne 0) {
    throw "Pipeline task failed with exit code $($container.exitCode): $($container.reason)"
}

$manifest = Invoke-AwsJson @("s3api", "head-object", "--bucket", $bucket, "--key", "curated/latest/manifest.json")
$objects = Invoke-AwsJson @("s3api", "list-objects-v2", "--bucket", $bucket, "--prefix", "curated/latest/")
$objectCount = @($objects.Contents).Count
if ($objectCount -lt 10) {
    throw "Artifact verification failed: expected at least 10 latest objects, found $objectCount."
}
Write-Host "Pipeline succeeded. Latest manifest: $($manifest.LastModified)"
Write-Host "Verified $objectCount objects under s3://$bucket/curated/latest/."
Write-Host "The recurring schedule remains disabled."

Remove-Item Env:AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
Remove-Item Env:AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
Remove-Item Env:AWS_SESSION_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:AWS_DEFAULT_REGION -ErrorAction SilentlyContinue
Remove-Item Env:AWS_REGION -ErrorAction SilentlyContinue
