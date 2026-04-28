data "google_client_config" "default" {}

data "google_container_cluster" "this" {
  name     = var.cluster_name
  location = var.region
}

# ─── Artifact Registry ──────────────────────────────────────────────────────
resource "google_artifact_registry_repository" "server" {
  repository_id = var.name_prefix
  location      = var.region
  format        = "DOCKER"
  description   = "Mantis Holo3 server images"

  cleanup_policies {
    id     = "keep-last-10"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
  cleanup_policy_dry_run = false
}

# ─── Secret Manager ─────────────────────────────────────────────────────────
locals {
  secrets = [
    "anthropic_api_key",
    "proxy_url",
    "proxy_user",
    "proxy_pass",
    "mantis_api_token",
  ]
}

resource "google_secret_manager_secret" "app" {
  for_each  = toset(local.secrets)
  secret_id = "${var.name_prefix}-${each.value}"

  replication {
    auto {}
  }
}

# ─── Filestore (RWX run state) ──────────────────────────────────────────────
resource "google_filestore_instance" "data" {
  name     = "${var.name_prefix}-data"
  location = var.zone
  tier     = "STANDARD"

  file_shares {
    name        = "data"
    capacity_gb = var.filestore_capacity_gb
  }

  networks {
    network = "default"
    modes   = ["MODE_IPV4"]
  }
}

# ─── GPU node pool ──────────────────────────────────────────────────────────
resource "google_container_node_pool" "gpu" {
  name     = "${var.name_prefix}-gpu"
  cluster  = var.cluster_name
  location = var.region

  node_locations = [var.zone]

  initial_node_count = var.gpu_node_initial

  autoscaling {
    min_node_count = var.gpu_node_min
    max_node_count = var.gpu_node_max
  }

  node_config {
    machine_type = var.gpu_machine_type
    spot         = var.use_spot_gpus
    disk_size_gb = 200
    disk_type    = "pd-ssd"

    guest_accelerator {
      type  = var.gpu_accelerator_type
      count = 1

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
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

    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ─── Workload Identity ServiceAccount ───────────────────────────────────────
resource "google_service_account" "app" {
  account_id   = "${var.name_prefix}-app"
  display_name = "Mantis Holo3 server"
}

resource "google_secret_manager_secret_iam_member" "app_access" {
  for_each  = google_secret_manager_secret.app
  project   = var.project_id
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

# Bind the GSA → KSA (Workload Identity).
# The KSA `default/mantis-holo3-server` must exist (created by deployment.yaml).
resource "google_service_account_iam_member" "app_workload_identity" {
  service_account_id = google_service_account.app.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[default/mantis-holo3-server]"
}

# ─── Outputs ────────────────────────────────────────────────────────────────
output "image_uri" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${var.name_prefix}/mantis-holo3-server:${var.image_tag}"
}

output "filestore_ip" {
  value = google_filestore_instance.data.networks[0].ip_addresses[0]
}

output "filestore_share_name" {
  value = google_filestore_instance.data.file_shares[0].name
}

output "app_gsa_email" {
  description = "Annotate the KSA with iam.gke.io/gcp-service-account = this."
  value       = google_service_account.app.email
}

output "secret_ids" {
  value = { for k, s in google_secret_manager_secret.app : k => s.secret_id }
}
