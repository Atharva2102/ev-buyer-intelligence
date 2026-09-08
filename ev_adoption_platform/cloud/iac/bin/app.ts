#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { EvDataPipelineStack } from "../lib/ev-data-pipeline-stack";

const app = new cdk.App();

new EvDataPipelineStack(app, "EvBuyerIntelligenceDataPipeline", {
  description: "Scheduled EV buyer analytics pipeline with S3, ECS Fargate, ECR, EventBridge, and CloudWatch",
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? "us-east-1"
  }
});
