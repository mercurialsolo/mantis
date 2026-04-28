data "aws_caller_identity" "current" {}

locals {
  account_id   = data.aws_caller_identity.current.account_id
  ecr_repo_url = "${local.account_id}.dkr.ecr.${var.region}.amazonaws.com/${var.name_prefix}-server"
}

# ─── ECR ────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "server" {
  name                 = "${var.name_prefix}-server"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "server" {
  repository = aws_ecr_repository.server.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ─── Secrets Manager ────────────────────────────────────────────────────────
locals {
  secrets = [
    "anthropic_api_key",
    "proxy_url",
    "proxy_user",
    "proxy_pass",
    "mantis_api_token",
  ]
}

resource "aws_secretsmanager_secret" "app" {
  for_each = toset(local.secrets)
  name     = "${var.name_prefix}/${each.value}"

  recovery_window_in_days = 0  # delete-immediately on terraform destroy (dev convenience; tighten for prod)
}

# Values are populated out-of-band via `aws secretsmanager put-secret-value`.

# ─── EFS for run state ──────────────────────────────────────────────────────
resource "aws_efs_file_system" "data" {
  creation_token   = "${var.name_prefix}-data"
  encrypted        = true
  throughput_mode  = "elastic"
  performance_mode = "generalPurpose"
  tags = {
    Name = "${var.name_prefix}-data"
  }
}

resource "aws_efs_mount_target" "data" {
  for_each        = toset(var.private_subnet_ids)
  file_system_id  = aws_efs_file_system.data.id
  subnet_id       = each.value
  security_groups = [var.node_security_group_id]
}

# ─── GPU NodeGroup (managed) ────────────────────────────────────────────────
data "aws_eks_cluster" "this" {
  name = var.cluster_name
}

# IAM role for the node group
data "aws_iam_policy_document" "ng_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu_nodes" {
  name               = "${var.name_prefix}-gpu-nodes"
  assume_role_policy = data.aws_iam_policy_document.ng_assume.json
}

resource "aws_iam_role_policy_attachment" "gpu_nodes_worker" {
  role       = aws_iam_role.gpu_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "gpu_nodes_cni" {
  role       = aws_iam_role.gpu_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "gpu_nodes_ecr" {
  role       = aws_iam_role.gpu_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.cluster_name
  node_group_name = "${var.name_prefix}-gpu"
  node_role_arn   = aws_iam_role.gpu_nodes.arn
  subnet_ids      = var.private_subnet_ids
  ami_type        = "AL2_x86_64_GPU"
  instance_types  = [var.gpu_instance_type]
  capacity_type   = "ON_DEMAND"

  scaling_config {
    desired_size = var.gpu_node_desired
    min_size     = var.gpu_node_min
    max_size     = var.gpu_node_max
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  labels = {
    workload = "mantis-holo3"
    gpu      = "true"
  }

  tags = {
    Name = "${var.name_prefix}-gpu-ng"
  }
}

# ─── IAM for ServiceAccount (IRSA) — secrets read + EFS write ───────────────
data "aws_iam_policy_document" "irsa_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_eks_cluster.this.identity[0].oidc[0].issuer]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(data.aws_eks_cluster.this.identity[0].oidc[0].issuer, "https://", "")}:sub"
      values   = ["system:serviceaccount:default:mantis-holo3-server"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "${var.name_prefix}-app"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust.json
}

data "aws_iam_policy_document" "app_perms" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      for s in aws_secretsmanager_secret.app : s.arn
    ]
  }
}

resource "aws_iam_role_policy" "app" {
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.app_perms.json
}

# ─── Outputs ────────────────────────────────────────────────────────────────
output "ecr_repo_url" {
  value = aws_ecr_repository.server.repository_url
}

output "image_uri" {
  value = "${local.ecr_repo_url}:${var.image_tag}"
}

output "efs_id" {
  value = aws_efs_file_system.data.id
}

output "app_role_arn" {
  description = "Annotate the k8s ServiceAccount with eks.amazonaws.com/role-arn = this."
  value       = aws_iam_role.app.arn
}

output "secret_arns" {
  value = { for k, s in aws_secretsmanager_secret.app : k => s.arn }
}
