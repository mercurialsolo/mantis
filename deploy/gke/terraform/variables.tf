variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the GPU node pool (must be a zone where the GPU type is available)."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "Existing GKE Standard cluster name."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resources + Secret Manager keys."
  type        = string
  default     = "mantis-prod"
}

variable "image_tag" {
  description = "Container image tag in Artifact Registry."
  type        = string
}

variable "gpu_machine_type" {
  description = "GCE machine type for the GPU node pool. a2-highgpu-1g = 1× A100 40GB."
  type        = string
  default     = "a2-highgpu-1g"
}

variable "gpu_accelerator_type" {
  description = "GPU accelerator. Options: nvidia-tesla-a100, nvidia-a100-80gb, nvidia-h100-80gb, nvidia-l4."
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "gpu_node_min" {
  type    = number
  default = 0
}

variable "gpu_node_max" {
  type    = number
  default = 2
}

variable "gpu_node_initial" {
  description = "Initial GPU node count."
  type        = number
  default     = 1
}

variable "use_spot_gpus" {
  description = "Use Spot GPUs (cheaper, preemptible). Recommended off for production."
  type        = bool
  default     = false
}

variable "filestore_capacity_gb" {
  description = "Filestore capacity in GB. Minimum 1024 for Standard tier."
  type        = number
  default     = 1024
}
