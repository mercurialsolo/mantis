# GKE

Self-hosted on GKE Standard (not Autopilot — Autopilot doesn't yet support custom GPU node pools with the taint config Mantis needs). Full Terraform + k8s manifests + runbook live at [`deploy/gke/`](https://github.com/mercurialsolo/mantis/tree/main/deploy/gke).

## Architecture

```
GCLB ingress (HTTPS, managed cert)
   ↓
GKE Standard cluster
  • GPU NodePool (a2-highgpu-1g — 1× A100 40 GB)
  • System NodePool
   ↓
Filestore (1 TB Standard, RWX — runs, checkpoints, profiles, recordings)
Artifact Registry (mantis-holo3-server image)
Secret Manager (anthropic_api_key, proxy_*, mantis_api_token)
Secret Manager CSI driver → in-pod tmpfs → envFrom Secret
```

## Footprint

| Resource | Type | Cost (us-central1) |
|---|---|---|
| GPU node | `a2-highgpu-1g` (A100 40 GB) | ~$3.67/hr on-demand |
| | `a2-ultragpu-1g` (A100 80 GB) | ~$4.61/hr (more headroom) |
| | `a3-highgpu-1g` (H100 80 GB) | ~$11.06/hr (high throughput) |
| | `a2-highgpu-1g` Spot | ~$1.10/hr (preemptible) |
| Filestore Standard | 1 TB | ~$200/month fixed |
| GCLB | LB + per-rule | ~$18/month + traffic |

For Holo3 Q8_0 (~34 GB VRAM), `a2-highgpu-1g` (A100 40 GB) is the sweet spot. Use Spot if you can tolerate preemption.

## End-to-end deploy

The detailed runbook is in [`deploy/gke/README.md`](https://github.com/mercurialsolo/mantis/blob/main/deploy/gke/README.md). High-level:

1. **Enable required APIs**
   ```bash
   gcloud services enable container.googleapis.com \
     artifactregistry.googleapis.com \
     secretmanager.googleapis.com
   ```

2. **Install cluster add-ons**
   - [Secret Manager CSI driver](https://github.com/GoogleCloudPlatform/secrets-store-csi-driver-provider-gcp) for Secret Manager → tmpfs
   - GPU device plugin DaemonSet (or auto-install on node pool creation)

3. **Build + push image to Artifact Registry**
   ```bash
   docker build --platform linux/amd64 -f docker/server.Dockerfile \
     -t "$REGION-docker.pkg.dev/$PROJECT/$REPO/mantis-holo3-server:$(git rev-parse --short HEAD)" .
   docker push "..."
   ```

4. **Provision infra**
   ```bash
   cd deploy/gke/terraform
   terraform init
   terraform apply -var "project_id=$PROJECT" \
                   -var "cluster_name=mantis-prod" \
                   -var "image_tag=$(git rev-parse --short HEAD)" \
                   -var "use_spot_gpus=false"
   ```

5. **Populate Secret Manager**
   ```bash
   for k in anthropic_api_key proxy_url proxy_user proxy_pass mantis_api_token; do
     read -rp "$k: " v
     printf "%s" "$v" | gcloud secrets versions add "mantis-prod-$k" --data-file=-
   done
   ```

6. **Pre-warm Holo3 weights**
   ```bash
   kubectl apply -f deploy/gke/k8s/init-holo3-weights.yaml
   ```

7. **Deploy the workload**
   ```bash
   kubectl apply -f deploy/gke/k8s/secret-provider-class.yaml
   kubectl apply -f deploy/gke/k8s/deployment.yaml
   kubectl apply -f deploy/gke/k8s/service.yaml
   kubectl apply -f deploy/gke/k8s/ingress.yaml
   ```

8. **Smoke-test**
   ```bash
   HOST=$(kubectl get ingress mantis-holo3-server \
     -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   TOK=$(gcloud secrets versions access latest --secret=mantis-prod-mantis_api_token)
   curl -fsS -X POST "https://$HOST/v1/predict" \
     -H "X-Mantis-Token: $TOK" \
     -d '{"detached": true, "micro": "plans/boattrader/extract_url_filtered_3listings.json", "state_key": "smoke"}'
   ```

## Operational notes

- **Spot pricing:** set `use_spot_gpus = true` for ~70 % savings. Plan for occasional preemption — runs survive a replica restart because state lives on Filestore.
- **Workload Identity:** the Terraform creates a GSA bound to `default/mantis-holo3-server` KSA so the pod reads Secret Manager without a JSON key on disk.
- **Filestore is fixed-cost:** 1 TB Standard is $200/month regardless of usage. For dev / lower-volume, drop to the minimum (`capacity_gb = 1024` is the floor for Standard).
- **Region:** A100s are scarce in some regions; `us-central1`, `us-west1`, and `europe-west4` are the safest bets.

## Status

Like the AWS path, this is a starter Terraform — tested locally with `terraform validate` but not deployed end-to-end against a live GCP project on this branch. Review for your VPC/IAM before `terraform apply`.

## See also

- [`deploy/gke/README.md`](https://github.com/mercurialsolo/mantis/blob/main/deploy/gke/README.md) — full runbook
- [`deploy/gke/k8s/`](https://github.com/mercurialsolo/mantis/tree/main/deploy/gke/k8s) — manifests
- [Tenant keys](../operations/tenant-keys.md)
