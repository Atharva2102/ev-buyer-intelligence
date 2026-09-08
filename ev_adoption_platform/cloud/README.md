# Cloud Data Pipeline

This directory contains a deployable AWS batch pipeline with optional Snowflake loading. The recurring schedule is disabled by default so synthesizing or deploying the stack does not immediately start billable Fargate runs.

## Components

- `iac/`: AWS CDK stack for S3, ECR, ECS Fargate, EventBridge, IAM, VPC, and CloudWatch.
- `pipeline/`: non-root Python container that builds Parquet marts and a portable DuckDB warehouse.
- `scripts/upload_inputs.py`: validates local source schemas and uploads canonical objects to S3.
- `tests/`: transformation and contract tests.

## Provision AWS Resources

Prerequisites: Node.js 20+, Docker, AWS CLI v2, and AWS credentials with CDK deployment permissions.

```powershell
cd ev_adoption_platform\cloud\iac
npm install
npx cdk bootstrap
npm run build
npm run synth
npm run deploy
```

Record the stack outputs. They include the data bucket, ECR URI, ECS cluster, task definition, public subnets, and security group.

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
