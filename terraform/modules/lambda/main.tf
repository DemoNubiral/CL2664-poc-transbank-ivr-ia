data "aws_caller_identity" "current" {}

## 
locals {
  unique_id = substr(md5("${var.Region}-${var.Environment}-${data.aws_caller_identity.current.account_id}"), 0, 8)
}

## Crear la función Lambda
resource "aws_lambda_function" "lambda_function" {
  function_name = "${var.lambda_function_name}-${var.Project}-${local.unique_id}"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = var.lambda_handler
  runtime       = var.lambda_runtime

  # Usar el archivo ZIP generado
  filename         = "${path.module}/code/${var.lambda_function_name}.zip"

  # dynamic "vpc_config" {
  #   for_each = var.use_vpc ? [1] : []
  #   content {
  #     subnet_ids         = var.subnet_ids
  #     security_group_ids = var.security_group_ids
  #   }
  # }

  depends_on = [
    aws_iam_role_policy.lambda_policy
  ]

  tags = {
    Name      = "${var.lambda_function_name}-${var.Project}-${local.unique_id}"
  }

}

## Role y Política de permisos para Lambda
resource "aws_iam_role" "lambda_execution_role" {
  ##name = replace("${var.Environment}-lambda-execution-role", "/[^a-zA-Z0-9+=,.@_-]/", "-")
  name = "lambda-execution-role-${var.Project}-${local.unique_id}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}


resource "aws_iam_role_policy" "lambda_policy" {
  name   = "lambda-policy-${var.Project}-${local.unique_id}"
  role   = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version: "2012-10-17",
    Statement: [
      {
        Sid: "LambdaLogging",
        Action: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect: "Allow",
        Resource: "arn:aws:logs:*:*:*"
      },
      {
        Sid: "Invokelambda",
        Action: [
          "lambda:InvokeFunction"
        ],
        Effect: "Allow",
        Resource: "arn:aws:lambda:*:*:function:*"
      }
    ]
  })
}
