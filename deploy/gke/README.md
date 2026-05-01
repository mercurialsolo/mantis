# GKE deployment

Self-host the Mantis Holo3 server on Google Kubernetes Engine using a GPU
node pool, Artifact Registry for the image, Secret Manager for runtime
credentials, Filestore for shared run state, and a GCLB ingress for HTTPS.

> **Status:** the Terraform here is a starter. It has not been applied
> against a live GCP project from this branch — review for your VPC/IAM
> before `terraform apply`. The k8s manifests have been validated only with
> `kubectl apply --dry-run=server`.

## Architecture

```
         ┌───────────────────────────┐
         │  GCLB Ingress (HTTPS)     │
         │  managed cert             │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  GKE cluster (Standard)   │
         │  ─────────────────────    │
         │  GPU node pool            │  ← a2-highgpu-1g (A100 40GB)
         │  ─────────────────────    │
         │  System node pool         │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  Filestore (RWX, 1 TB)    │  ← /workspace/mantis-data
         │  GCS bucket (Holo3 GGUF)  │  ← /models/holo3 init-job source
         │  Secret Manager           │  ← env via Secret Manager CSI
         │  Artifact Registry        │  ← container image
         └───────────────────────────┘
```

## Instance type

Holo3-35B-A3B GGUF Q8_0 needs ~34 GB VRAM. Recommended:

| Machine type            | GPU            | VRAM   | Notes                     |
|-------------------------|----------------|--------|---------------------------|
| `g2-standard-8`         | L4             | 24 GB  | Too small for Q8_0        |
| `g2-standard-12`        | L4             | 24 GB  | Same                      |
| `n1-standard-8 + L4`    | L4             | 24 GB  | Same                      |
| `a2-highgpu-1g`         | A100 40 GB     | 40 GB  | Sweet spot                |
| `a2-ultragpu-1g`        | A100 80 GB     | 80 GB  | Headroom for batches      |
| `a3-highgpu-1g`         | H100 80 GB     | 80 GB  | High-throughput option    |

For a single replica, **`a2-highgpu-1g`** (A100 40 GB) is the cheapest viable
choice (~$3.67/hr in us-central1).

## One-time prerequisites

1. `gcloud auth login`, `gcloud config set project <ID>`.
2. Enable APIs: `gcloud services enable container.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com`.
3. Install [Secret Manager CSI driver](https://github.com/GoogleCloudPlatform/secrets-store-csi-driver-provider-gcp) in the cluster.
4. Install the [GKE GPU driver DaemonSet](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) (or use the auto-install option on node pool creation).
5. Have an existing GKE cluster (Standard, not Autopilot — Autopilot does not yet support custom GPU node pools with our taint config).

## 1. Build + push the image

```bash
PROJECT=$(gcloud config get-value project)
REGION=us-central1
REPO=mantis-prod
IMAGE=mantis-holo3-server

gcloud artifacts repositories describe "$REPO" --location="$REGION" \
  || gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"

gcloud auth configure-docker "$REGION-docker.pkg.dev"

docker build --platform linux/amd64 \
  -f docker/server.Dockerfile \
  -t "$REGION-docker.pkg.dev/$PROJECT/$REPO/$IMAGE:$(git rev-parse --short HEAD)" .

docker push "$REGION-docker.pkg.dev/$PROJECT/$REPO/$IMAGE:$(git rev-parse --short HEAD)"
```

## 2. Provision infra (Terraform)

```bash
cd deploy/gke/terraform
terraform init
terraform apply \
  -var "project_id=$(gcloud config get-value project)" \
  -var "region=us-central1" \
  -var "cluster_name=mantis-prod" \
  -var "image_tag=$(git rev-parse --short HEAD)"
```

This creates: Artifact Registry repo, Filestore instance, the 5 Secret
Manager secrets (empty — populate manually), a GPU node pool on the existing
cluster, an IAM service account for Workload Identity. It does **not**
create the GKE control plane — bring your own cluster.

## 3. Populate secrets

```bash
PROJECT=$(gcloud config get-value project)
for k in anthropic_api_key proxy_url proxy_user proxy_pass mantis_api_token; do
  read -rp "$k: " v
  printf "%s" "$v" | gcloud secrets versions add "mantis-prod-$k" --data-file=- --project="$PROJECT"
done
```

## 4. Pre-warm the Holo3 weights

```bash
kubectl apply -f deploy/gke/k8s/init-holo3-weights.yaml
kubectl logs -f job/init-holo3-weights
```

## 5. Deploy the workload

```bash
kubectl apply -f deploy/gke/k8s/secret-provider-class.yaml   # Secret Manager → tmpfs
kubectl apply -f deploy/gke/k8s/deployment.yaml
kubectl apply -f deploy/gke/k8s/service.yaml
kubectl apply -f deploy/gke/k8s/ingress.yaml                # GCLB + managed cert
kubectl rollout status deploy/mantis-holo3-server
```

## 6. Smoke-test

```bash
PROJECT=$(gcloud config get-value project)
TOK=$(gcloud secrets versions access latest --secret=mantis-prod-mantis_api_token --project="$PROJECT")
HOST=$(kubectl get ingress mantis-holo3-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

curl -fsS -X POST "https://$HOST/predict" \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/example/extract_listings.json",
    "state_key": "gke-smoke",
    "resume_state": false,
    "max_cost": 2,
    "max_time_minutes": 20
  }'
```

Expected: `{"status": "queued", "run_id": "...", ...}`. Poll with
`{"action":"status","run_id":"..."}` against the same endpoint.

## Cost guardrails

- The Deployment defaults to 1 replica; idle GPU costs accumulate. For
  bursty traffic, scale-to-zero with [KEDA](https://keda.sh/) or use
  preemptible/spot GPUs (set `node_pool.spot = true` in `main.tf`).
- Artifact Registry has cleanup policies via Terraform (`cleanup_policies`)
  — keeps last 10 images in the starter.
- Filestore Standard is $0.20/GB/month — the 1 TB default = $200/month
  fixed. Scale down or switch to Basic if you don't need throughput.

## Pre-existing limitations carried over

Same as AWS: Holo3 occasionally emits clicks instead of scrolls on detail
pages — the `MicroPlanRunner` already has a scroll-fail-as-success
fallback. The default residential proxy is geo-targeted to Miami via IPRoyal.
