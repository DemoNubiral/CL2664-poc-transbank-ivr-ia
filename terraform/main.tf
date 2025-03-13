## Módulo Lambda
module "lambda" {
  source               = "./modules/lambda"

  Region               = var.Region
  Environment          = var.Environment
  Project              = var.Project

  lambda_function_name = var.lambda_function_name
  lambda_handler       = var.lambda_handler
  lambda_runtime       = var.lambda_runtime
  lambda_batch_size    = var.lambda_batch_size
  
}
