from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
    aws_iam as iam,
    CfnOutput
)
from constructs import Construct
import os

def get_relative_path():
    current_file = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file)
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    return root_dir

def get_project_root():
    return os.path.join(get_relative_path())

class ChatbotCdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # create the Lambda Function from the Docker Image
        lambda_function = _lambda.DockerImageFunction(
            self, "RAGChatbotLambda",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=os.path.join(get_project_root(), "src"),
                exclude=["chatbot_cdk", "cdk.out", "venv", ".venv"]
            ),
            architecture=_lambda.Architecture.X86_64,
            memory_size=1024,
            timeout=Duration.seconds(30),
            environment={
                "ENVIRONMENT": "PRODUCTION"
            }
        )

        functionUrl = lambda_function.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE,
        )

        lambda_function.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess")
        )

        # output the API endpoint
        CfnOutput(self, "API Endpoint", value=functionUrl.url)