variable "region" {
  description = "AWS region for ECR / EFS / Secrets Manager."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Existing EKS cluster name. This module does NOT create the control plane."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names + Secrets Manager paths."
  type        = string
  default     = "mantis-prod"
}

variable "image_tag" {
  description = "Container image tag in ECR (typically the short git SHA)."
  type        = string
}

variable "gpu_instance_type" {
  description = "EC2 instance for the GPU node group. g6e.2xlarge is the cheapest option that fits Holo3 Q8_0."
  type        = string
  default     = "g6e.2xlarge"
}

variable "gpu_node_min" {
  description = "Min GPU nodes."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  description = "Max GPU nodes."
  type        = number
  default     = 2
}

variable "gpu_node_desired" {
  description = "Desired GPU nodes (set to 1 to start serving)."
  type        = number
  default     = 1
}

variable "vpc_id" {
  description = "VPC the EKS cluster lives in (used for EFS mount targets)."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for EFS mount targets and node groups."
  type        = list(string)
}

variable "node_security_group_id" {
  description = "SG attached to EKS worker nodes (allows EFS NFS traffic)."
  type        = string
}
