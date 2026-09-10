import * as cdk from "aws-cdk-lib";
import * as apprunner from "aws-cdk-lib/aws-apprunner";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export class EvDataPipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const contextBoolean = (name: string): boolean => {
      const value = this.node.tryGetContext(name);
      return value === true || String(value).toLowerCase() === "true";
    };

    const scheduleEnabled = contextBoolean("scheduleEnabled");
    const deployApiService = contextBoolean("deployApiService");
    const deployApiLambda = contextBoolean("deployApiLambda");
    const apiImageTag = String(this.node.tryGetContext("apiImageTag") ?? "latest").trim();
    const scheduleExpression = String(
      this.node.tryGetContext("scheduleExpression") ?? "cron(0 6 * * ? *)"
    );
    const enableSnowflake = contextBoolean("enableSnowflake");
    const snowflakeSecretArn = String(
      this.node.tryGetContext("snowflakeSecretArn") ?? ""
    ).trim();
    const frontendOrigins = String(
      this.node.tryGetContext("frontendOrigins")
      ?? "https://atharva2102.github.io,http://localhost:3000,http://127.0.0.1:3000"
    ).trim();
    if (enableSnowflake && !snowflakeSecretArn) {
      throw new Error("snowflakeSecretArn is required when enableSnowflake=true");
    }

    const dataBucket = new s3.Bucket(this, "DataBucket", {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: true,
      lifecycleRules: [
        {
          id: "expire-noncurrent-versions",
          noncurrentVersionExpiration: cdk.Duration.days(30)
        },
        {
          id: "archive-pipeline-runs",
          prefix: "curated/runs/",
          transitions: [
            { storageClass: s3.StorageClass.INFREQUENT_ACCESS, transitionAfter: cdk.Duration.days(30) }
          ]
        }
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN
    });

    const repository = new ecr.Repository(this, "PipelineRepository", {
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 10 }],
      removalPolicy: cdk.RemovalPolicy.RETAIN
    });

    const apiRepository = new ecr.Repository(this, "ApiRepository", {
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 10 }],
      removalPolicy: cdk.RemovalPolicy.RETAIN
    });

    const scoringEvents = new dynamodb.Table(this, "ScoringEvents", {
      partitionKey: { name: "stream_key", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "occurred_at_event", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.DEFAULT,
      timeToLiveAttribute: "expires_at",
      removalPolicy: cdk.RemovalPolicy.RETAIN
    });

    const vpc = new ec2.Vpc(this, "PipelineVpc", {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 }
      ]
    });
    vpc.addGatewayEndpoint("S3Endpoint", {
      service: ec2.GatewayVpcEndpointAwsService.S3
    });

    const cluster = new ecs.Cluster(this, "PipelineCluster", {
      vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED
    });

    const logGroup = new logs.LogGroup(this, "PipelineLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, "PipelineTask", {
      cpu: 1024,
      memoryLimitMiB: 4096,
      runtimePlatform: {
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX
      }
    });

    taskDefinition.addContainer("PipelineContainer", {
      image: ecs.ContainerImage.fromEcrRepository(repository, "latest"),
      logging: ecs.LogDrivers.awsLogs({ logGroup, streamPrefix: "etl" }),
      environment: {
        DATA_BUCKET: dataBucket.bucketName,
        RAW_PREFIX: "raw",
        CURATED_PREFIX: "curated",
        SCORES_KEY: "raw/scores.csv",
        ENABLE_SNOWFLAKE: String(enableSnowflake),
        SNOWFLAKE_SECRET_ARN: snowflakeSecretArn,
        PIPELINE_VERSION: "0.2.0"
      }
    });

    dataBucket.grantRead(taskDefinition.taskRole, "raw/*");
    dataBucket.grantRead(taskDefinition.taskRole, "curated/runs/*");
    dataBucket.grantPut(taskDefinition.taskRole, "curated/*");
    repository.grantPull(taskDefinition.executionRole!);
    if (enableSnowflake) {
      taskDefinition.taskRole.addToPrincipalPolicy(
        new iam.PolicyStatement({
          actions: ["secretsmanager:GetSecretValue"],
          resources: [snowflakeSecretArn]
        })
      );
    }

    const securityGroup = new ec2.SecurityGroup(this, "PipelineSecurityGroup", {
      vpc,
      description: "Outbound-only access for the EV analytics batch task",
      allowAllOutbound: true
    });

    const schedule = new events.Rule(this, "PipelineSchedule", {
      description: "Runs the EV buyer analytics batch pipeline",
      enabled: scheduleEnabled,
      schedule: events.Schedule.expression(scheduleExpression)
    });

    schedule.addTarget(
      new targets.EcsTask({
        cluster,
        taskDefinition,
        taskCount: 1,
        assignPublicIp: true,
        subnetSelection: { subnetType: ec2.SubnetType.PUBLIC },
        securityGroups: [securityGroup]
      })
    );

    const apiEcrAccessRole = new iam.Role(this, "ApiEcrAccessRole", {
      assumedBy: new iam.ServicePrincipal("build.apprunner.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSAppRunnerServicePolicyForECRAccess")
      ]
    });

    const apiInstanceRole = new iam.Role(this, "ApiInstanceRole", {
      assumedBy: new iam.ServicePrincipal("tasks.apprunner.amazonaws.com")
    });
    dataBucket.grantRead(apiInstanceRole, "curated/latest/ev_adoption.duckdb");
    scoringEvents.grantReadWriteData(apiInstanceRole);
    apiInstanceRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ["dynamodb:DescribeTable"],
        resources: [scoringEvents.tableArn]
      })
    );

    if (deployApiService) {
      const apiAutoScaling = new apprunner.CfnAutoScalingConfiguration(this, "ApiAutoScaling", {
        autoScalingConfigurationName: "ev-buyer-api-cost-controlled",
        minSize: 1,
        maxSize: 2,
        maxConcurrency: 80
      });

      const apiService = new apprunner.CfnService(this, "ApiService", {
        serviceName: "ev-buyer-intelligence-api",
        autoScalingConfigurationArn: apiAutoScaling.attrAutoScalingConfigurationArn,
        sourceConfiguration: {
          autoDeploymentsEnabled: false,
          authenticationConfiguration: {
            accessRoleArn: apiEcrAccessRole.roleArn
          },
          imageRepository: {
            imageIdentifier: `${apiRepository.repositoryUri}:${apiImageTag}`,
            imageRepositoryType: "ECR",
            imageConfiguration: {
              port: "8000",
              runtimeEnvironmentVariables: [
                { name: "AWS_REGION", value: this.region },
                { name: "AWS_DEFAULT_REGION", value: this.region },
                { name: "DATA_BUCKET", value: dataBucket.bucketName },
                { name: "WAREHOUSE_S3_KEY", value: "curated/latest/ev_adoption.duckdb" },
                { name: "WAREHOUSE_PATH", value: "/app/warehouse/ev_adoption.duckdb" },
                { name: "MODEL_DIR", value: "/app/models" },
                { name: "SCORING_EVENTS_TABLE", value: scoringEvents.tableName },
                { name: "CORS_ORIGINS", value: frontendOrigins }
              ]
            }
          }
        },
        instanceConfiguration: {
          cpu: "1 vCPU",
          memory: "2 GB",
          instanceRoleArn: apiInstanceRole.roleArn
        },
        healthCheckConfiguration: {
          protocol: "HTTP",
          path: "/health",
          interval: 10,
          timeout: 5,
          healthyThreshold: 1,
          unhealthyThreshold: 5
        }
      });
      apiService.addResourceDependency(apiAutoScaling);
      new cdk.CfnOutput(this, "ApiServiceUrl", {
        value: `https://${apiService.attrServiceUrl}`
      });
      new cdk.CfnOutput(this, "ApiServiceArn", { value: apiService.attrServiceArn });
    }

    if (deployApiLambda) {
      const apiFunction = new lambda.DockerImageFunction(this, "ApiFunction", {
        functionName: "ev-buyer-intelligence-api",
        code: lambda.DockerImageCode.fromEcr(apiRepository, { tagOrDigest: apiImageTag }),
        architecture: lambda.Architecture.X86_64,
        memorySize: 2048,
        timeout: cdk.Duration.seconds(90),
        environment: {
          DATA_BUCKET: dataBucket.bucketName,
          WAREHOUSE_S3_KEY: "curated/latest/ev_adoption.duckdb",
          WAREHOUSE_PATH: "/tmp/ev_adoption.duckdb",
          MODEL_DIR: "/var/task/models",
          SCORING_EVENTS_TABLE: scoringEvents.tableName,
          CORS_ORIGINS: frontendOrigins
        }
      });
      dataBucket.grantRead(apiFunction, "curated/latest/ev_adoption.duckdb");
      scoringEvents.grantReadWriteData(apiFunction);
      apiFunction.addToRolePolicy(
        new iam.PolicyStatement({
          actions: ["dynamodb:DescribeTable"],
          resources: [scoringEvents.tableArn]
        })
      );

      const functionUrl = apiFunction.addFunctionUrl({
        authType: lambda.FunctionUrlAuthType.NONE
      });
      new cdk.CfnOutput(this, "ApiLambdaUrl", { value: functionUrl.url });
      new cdk.CfnOutput(this, "ApiLambdaArn", { value: apiFunction.functionArn });
    }

    new cdk.CfnOutput(this, "DataBucketName", { value: dataBucket.bucketName });
    new cdk.CfnOutput(this, "PipelineRepositoryUri", { value: repository.repositoryUri });
    new cdk.CfnOutput(this, "ApiRepositoryUri", { value: apiRepository.repositoryUri });
    new cdk.CfnOutput(this, "ScoringEventsTableName", { value: scoringEvents.tableName });
    new cdk.CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new cdk.CfnOutput(this, "TaskDefinitionArn", { value: taskDefinition.taskDefinitionArn });
    new cdk.CfnOutput(this, "ScheduleName", { value: schedule.ruleName });
    new cdk.CfnOutput(this, "ScheduleEnabled", { value: String(scheduleEnabled) });
    new cdk.CfnOutput(this, "ApiServiceEnabled", { value: String(deployApiService) });
    new cdk.CfnOutput(this, "ApiLambdaEnabled", { value: String(deployApiLambda) });
    new cdk.CfnOutput(this, "SnowflakeEnabled", { value: String(enableSnowflake) });
    new cdk.CfnOutput(this, "PublicSubnetIds", {
      value: vpc.publicSubnets.map((subnet) => subnet.subnetId).join(",")
    });
    new cdk.CfnOutput(this, "SecurityGroupId", { value: securityGroup.securityGroupId });

    cdk.Tags.of(this).add("Project", "EVBuyerIntelligence");
    cdk.Tags.of(this).add("ManagedBy", "AWS-CDK");
    cdk.Tags.of(this).add("Workload", "AnalyticsPipeline");
  }
}
