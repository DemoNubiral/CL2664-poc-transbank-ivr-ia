# provider.tf

## Necesario para el generated_backend
terraform {
  backend "s3" {}
}

provider "aws" {
  region = var.Region

  default_tags {
    tags = {
      Environment = var.Environment
      project     = var.Project
      Team        = var.Team
      owner       = var.owner
      createdBy   = var.createdBy
      deadline    = var.deadline
      "pod/coe"   = "ai"
    }
  }
}
