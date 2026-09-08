import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as iam from "aws-cdk-lib/aws-iam";
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
    const scheduleExpression = String(
      this.node.tryGetContext("scheduleExpression") ?? "cron(0 6 * * ? *)"
    );
    const enableSnowflake = contextBoolean("enableSnowflake");
    const snowflakeSecretArn = String(
      this.node.tryGetContext("snowflakeSecretArn") ?? ""
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
        PIPELINE_VERSION: "0.1.0"
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

    new cdk.CfnOutput(this, "DataBucketName", { value: dataBucket.bucketName });
    new cdk.CfnOutput(this, "PipelineRepositoryUri", { value: repository.repositoryUri });
    new cdk.CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new cdk.CfnOutput(this, "TaskDefinitionArn", { value: taskDefinition.taskDefinitionArn });
    new cdk.CfnOutput(this, "ScheduleName", { value: schedule.ruleName });
    new cdk.CfnOutput(this, "ScheduleEnabled", { value: String(scheduleEnabled) });
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
