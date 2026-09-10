# Cloud Data Pipeline and Serving API

This directory contains a deployable AWS batch pipeline with optional Snowflake loading. The recurring schedule is disabled by default so synthesizing or deploying the stack does not immediately start billable Fargate runs.

## Components

- `iac/`: AWS CDK stack for S3, ECR, ECS Fargate, DynamoDB, EventBridge, IAM, VPC, and CloudWatch.
- `pipeline/`: non-root Python container that builds Parquet marts and a portable DuckDB warehouse.
- `scripts/upload_inputs.py`: validates local source schemas and uploads canonical objects to S3.
- `tests/`: transformation and contract tests.
- `../api/Dockerfile.lambda`: production FastAPI image used by AWS Lambda.
- `../api/Dockerfile`: optional App Runner image retained for future use.

## Provision AWS Resources

Prerequisites: Node.js 20+, Docker, AWS CLI v2, and AWS credentials with CDK deployment permissions.

For a standalone AWS Free plan account, do not enable AWS Organizations merely to use IAM Identity Center: organization creation upgrades the account plan. Use an MFA-protected deployment role instead:

1. In AWS Budgets, create a `$20` monthly budget with actual and forecast notifications.
2. Create `ev-platform-bootstrap` as an IAM user with no console password.
3. Assign a virtual MFA device named `ev-platform-bootstrap` and create one access key.
4. Create `EVPlatformDeploymentRole`, temporarily attach `AdministratorAccess`, and trust only the bootstrap user when `aws:MultiFactorAuthPresent` is `true`.
5. Give the bootstrap user only `sts:AssumeRole` access to `arn:aws:iam::<account-id>:role/EVPlatformDeploymentRole`.

Use this role trust policy, replacing `<account-id>`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::<account-id>:user/ev-platform-bootstrap"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "Bool": {
          "aws:MultiFactorAuthPresent": "true"
        }
      }
    }
  ]
}
```

Attach only this inline policy to the bootstrap user, again replacing `<account-id>`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::<account-id>:role/EVPlatformDeploymentRole"
    }
  ]
}
```

Configure the source credentials locally; never paste or commit them:

```powershell
aws configure --profile ev-bootstrap
./cloud/scripts/configure_role_profile.ps1
aws sts get-caller-identity --profile ev-deploy
```

After the first deployment, replace `AdministratorAccess` with a least-privilege deployment policy based on the resources used, and delete or rotate the bootstrap access key.

```powershell
cd ev_adoption_platform\cloud\iac
npm install
npx cdk bootstrap
npm run build
npm run synth
npm run deploy
```

Record the stack outputs. They include the data bucket, ECR URI, ECS cluster, task definition, public subnets, and security group.

## Automated First Deployment

After the `ev-deploy` profile works, the deployment script synthesizes and deploys with scheduling and Snowflake disabled, builds and pushes the image, uploads validated inputs, runs one task, and verifies the latest S3 manifest. CDK bootstrap is a one-time operation; add `-Bootstrap` only for a new account and Region:

```powershell
cd ev_adoption_platform
./cloud/scripts/deploy_pipeline.ps1
./cloud/scripts/deploy_pipeline.ps1 -Bootstrap  # first deployment only
```

The retained score-file default is `submission_chris_lgbm_tuned_blend.csv`. Override it when the stronger TabM file is available:

```powershell
./cloud/scripts/deploy_pipeline.ps1 -ScoresPath C:\path\to\submission_tabm_colab.csv
```

## Upload Inputs

From `ev_adoption_platform/`:

```powershell
pip install boto3 pandas
python cloud\scripts\upload_inputs.py `
  --bucket <DataBucketName> `
  --train ..\kaggle_competition_workspace\data\train.csv `
  --test ..\kaggle_competition_workspace\data\test.csv `
  --original "..\kaggle_competition_workspace\data\OG Dataset\EV_Adoption_and_Range_Anxiety_Dataset.csv" `
  --scores ..\kaggle_competition_workspace\submissions\top_10\submission_tabm_colab.csv
```

The uploader writes `raw/train.csv`, `raw/test.csv`, `raw/original_ev_adoption.csv`, and `raw/scores.csv`, each with a SHA-256 metadata value.

The container is pinned to Debian Bookworm, installs available OS security updates at build time, runs as UID `10001`, and is scanned when pushed to ECR. Review unresolved upstream findings in ECR before treating an image as production-ready.

## Scoring Event Store

The stack provisions an on-demand DynamoDB table with a 30-day TTL for simulator and explicitly labeled demo-stream events. The API defaults to local SQLite; start it against DynamoDB with an MFA-backed temporary session:

```powershell
cd ev_adoption_platform
./cloud/scripts/start_api_aws.ps1
```

`/scoring-operations` aggregates stored events into request rate, latency, success, source, and adoption-band metrics. `/demo-score` samples a synthetic profile and labels the event `demo_stream`; `/predict` records user-triggered simulator activity.

## Deploy the Public API

The verified production path packages FastAPI as a Lambda container, adapts
ASGI requests with Mangum, and exposes a public Function URL. The function uses
IAM roles rather than embedded credentials, reads only the latest DuckDB object
from S3, and writes scoring events to DynamoDB:

```powershell
cd ev_adoption_platform
.\cloud\scripts\deploy_api_lambda.ps1
```

The function uses 2 GB of memory and a 90-second timeout. At cold start it
downloads `curated/latest/ev_adoption.duckdb` into `/tmp`, loads the LightGBM
serving artifact, and initializes the DynamoDB event store. The account-level
Lambda concurrency quota bounds concurrent executions.

To add the final frontend domain to CORS:

```powershell
.\cloud\scripts\deploy_api_lambda.ps1 `
  -FrontendOrigins "https://evbuyerintelligence.dev,https://atharva2102.github.io"
```

Successful deployment prints `ApiLambdaUrl`. Save that value as the GitHub
repository variable `NEXT_PUBLIC_API_BASE` before running the Pages workflow.
Do not enable the batch schedule as part of API deployment.

`deploy_api.ps1` retains the original App Runner deployment path. It is disabled
in the verified stack because App Runner service creation failed for this new
AWS account even when tested with AWS's own hello image. Lambda provides the
same public API contract with usage-based billing and no idle instance.

## Build and Push the Container

Run these commands from `ev_adoption_platform/`. Replace placeholders with CDK outputs and your AWS region/account.

```powershell
docker build -f cloud\pipeline\Dockerfile -t ev-buyer-pipeline .
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag ev-buyer-pipeline:latest <PipelineRepositoryUri>:latest
docker push <PipelineRepositoryUri>:latest
```

## Run Once

```powershell
aws ecs run-task `
  --cluster <ClusterName> `
  --task-definition <TaskDefinitionArn> `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-1>,<subnet-2>],securityGroups=[<SecurityGroupId>],assignPublicIp=ENABLED}"
```

Inspect logs under the CDK-created CloudWatch log group. Successful runs write:

```text
s3://<bucket>/curated/runs/<timestamp>/*.parquet
s3://<bucket>/curated/runs/<timestamp>/ev_adoption.duckdb
s3://<bucket>/curated/runs/<timestamp>/manifest.json
s3://<bucket>/curated/latest/*
```

## Enable Scheduling

Deploy with an explicit context flag after the image and raw objects exist:

```powershell
npx cdk deploy -c scheduleEnabled=true -c "scheduleExpression=cron(0 6 * * ? *)"
```

## Enable Snowflake

Create an AWS Secrets Manager secret named with the prefix `ev-buyer-intelligence/snowflake-`. Its JSON value must contain:

```json
{
  "account": "organization-account",
  "user": "pipeline_user",
  "password": "replace-me",
  "warehouse": "EV_PIPELINE_WH",
  "database": "EV_BUYER_INTELLIGENCE",
  "role": "EV_PIPELINE_ROLE"
}
```

Enable the loader through CDK context and redeploy:

```powershell
npx cdk deploy -c enableSnowflake=true -c snowflakeSecretArn=<secret-arn>
```

The loader creates `STAGING`, `SERVING`, and `MART` schemas and replaces pipeline-owned tables through Snowflake connector uploads.

## Local Tests

```powershell
cd ev_adoption_platform
pip install -r cloud\pipeline\requirements.txt
python -m unittest discover -s cloud\tests -v
```

No AWS resources are created by tests or `cdk synth`. AWS charges begin only after deployment and task execution.
