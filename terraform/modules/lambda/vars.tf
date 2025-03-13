## Variables

variable "Environment" {
  description = "Entorno de AWS"
  type        = string
}

variable "Region" {
  description = "Etiquetas para asignar al recurso."
  type        = string
}

variable "Project" {
  description = "Nombre del proyecto"
  type        = string
}

variable "lambda_function_name" {
  description = "Nombre de la función Lambda"
  type        = string
}

variable "lambda_handler" {
  description = "Handler de la función Lambda"
  type        = string
}

variable "lambda_runtime" {
  description = "Runtime a usar en la función Lambda"
  type        = string
}

variable "lambda_batch_size" {
  description = "Número de mensajes que procesará Lambda del trigger SQS"
  type        = number
}
