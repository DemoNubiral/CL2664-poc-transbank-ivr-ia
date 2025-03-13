## Variables
## Default Tags

variable "Region" {
  description = "Region de AWS"
  type        = string
}

variable "Environment" {
  description = "Entorno de AWS"
  type        = string
}

variable "Project" {
  description = "Nombre del proyecto"
  type        = string
}

variable "Team" {
  description = "Nombre del Pod"
  type        = string
}

variable "owner" {
  description = "Correo del creador"
  type        = string
}

variable "createdBy" {
  description = "Nombre del creador"
  type        = string
}

variable "deadline" {
  description = "Fecha limite"
  type        = string
}

## Module Lambda
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