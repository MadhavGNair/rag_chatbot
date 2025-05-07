import aws_cdk as core
import aws_cdk.assertions as assertions

from chatbot_cdk.chatbot_cdk_stack import ChatbotCdkStack

# example tests. To run these tests, uncomment this file along with the example
# resource in chatbot_cdk/chatbot_cdk_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = ChatbotCdkStack(app, "chatbot-cdk")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
