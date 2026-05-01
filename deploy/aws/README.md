# AWS deployment (EKS)

Self-host the Mantis Holo3 server on AWS using EKS + a single GPU node group.
This sketch is the practical minimum: ECR for the image, Secrets Manager for
runtime credentials, EFS for persistent run state, an Application Load
Balancer for HTTPS ingress.

> **Status:** the Terraform here is a starter. It has not been applied against
> a live AWS account from this branch — review for your VPC, IAM, and
> compliance needs before `terraform apply`. The k8s manifests have been
> validated only with `kubectl apply --dry-run=server`.

## Architecture

```
         ┌───────────────────────────┐
         │  ALB  (HTTPS, ACM cert)   │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  EKS cluster              │
         │  ─────────────────────    │
         │  GPU NodeGroup (g5/p4)    │  ← Holo3 inference + Chrome
         │  ─────────────────────    │
         │  CPU NodeGroup (m6i)      │  ← cluster add-ons, ALB ctrl
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  EFS (run state, leads)   │  ← /workspace/mantis-data
         │  S3 bucket (Holo3 GGUF)   │  ← /models/holo3 via init-job
         │  Secrets Manager          │  ← env via External Secrets Operator
         │  ECR (mantis-holo3-server)│  ← container image
         └───────────────────────────┘
```

## Instance type

Holo3-35B-A3B GGUF Q8_0 needs ~34 GB VRAM. Recommended:

| Instance         | GPU       | VRAM | Notes                         |
|------------------|-----------|------|-------------------------------|
| `g5.2xlarge`     | A10G      | 24 GB | Too small — quantize further or pick larger |
| `g6e.2xlarge`    | L40S      | 48 GB | Good fit; cheapest option that holds Q8_0   |
| `g5.12xlarge`    | 4× A10G   | 96 GB | Overkill for a single model   |
| `p4d.24xlarge`   | 8× A100   | 320 GB | Reserved-only; expensive      |
| `p5.48xlarge`    | 8× H100   | 640 GB | Reserved-only; expensive      |

For a single replica, **`g6e.2xlarge`** is the sweet spot (~$1.86/hr on-demand).

## One-time prerequisites

1. **Configure AWS CLI** and `kubectl`.
2. **EKS cluster** with the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) installed.
3. **AWS Load Balancer Controller** installed for the `alb` ingress class.
4. **EFS CSI driver** installed if you want PVC-backed run state.
5. **External Secrets Operator** installed (or roll your own Secrets Manager → env mapping).

## 1. Build + push the image

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO=mantis-holo3-server

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

docker build --platform linux/amd64 \
  -f docker/server.Dockerfile \
  -t "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$(git rev-parse --short HEAD)" .

docker push "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$(git rev-parse --short HEAD)"
```

## 2. Provision infra (Terraform)

```bash
cd deploy/aws/terraform
terraform init
terraform apply \
  -var "region=us-east-1" \
  -var "cluster_name=mantis-prod" \
  -var "image_tag=$(git rev-parse --short HEAD)"
```

This creates: ECR repo, EFS file system, secrets in Secrets Manager (with
empty values — populate manually), GPU + CPU node groups. It does **not**
create the EKS control plane itself — bring your own cluster (e.g., from the
`terraform-aws-modules/eks/aws` module) and pass its name in.

## 3. Populate secrets

```bash
PREFIX=mantis/prod
for k in anthropic_api_key proxy_url proxy_user proxy_pass mantis_api_token; do
  aws secretsmanager put-secret-value \
    --secret-id "$PREFIX/$k" \
    --secret-string "$(read -rp "$k: " v && echo "$v")"
done
```

## 4. Pre-warm the Holo3 weights

The image does not bake the GGUF (~34 GB). Run a one-time init job that
downloads from HuggingFace into the EFS-backed `/models/holo3` PVC:

```bash
kubectl apply -f deploy/aws/k8s/init-holo3-weights.yaml
kubectl logs -f job/init-holo3-weights
```

(See the YAML for the `huggingface-cli download` invocation.)

## 5. Deploy the workload

```bash
kubectl apply -f deploy/aws/k8s/external-secret.yaml   # Secrets Manager → k8s Secret
kubectl apply -f deploy/aws/k8s/deployment.yaml
kubectl apply -f deploy/aws/k8s/service.yaml
kubectl apply -f deploy/aws/k8s/ingress.yaml           # ALB ingress with ACM cert
kubectl rollout status deploy/mantis-holo3-server
```

## 6. Smoke-test

```bash
TOK=$(aws secretsmanager get-secret-value --secret-id mantis/prod/mantis_api_token --query SecretString --output text)
HOST=$(kubectl get ingress mantis-holo3-server -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

curl -fsS -X POST "https://$HOST/predict" \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/example/extract_listings.json",
    "state_key": "aws-eks-smoke",
    "resume_state": false,
    "max_cost": 2,
    "max_time_minutes": 20
  }'
```

Expected response: `{"status": "queued", "run_id": "...", ...}`. Poll with
`{"action":"status","run_id":"..."}` against the same endpoint.

## Cost guardrails

- The HPA in `deployment.yaml` scales 1 → 1 by default (single replica). Idle
  GPU costs add up — set `min_replicas: 0` and use a scale-to-zero strategy
  (KEDA / Karpenter) if your traffic is bursty.
- ECR lifecycle policy is in `terraform/main.tf` — keeps last 10 images.
- EFS throughput mode defaults to "elastic" (pay-per-use); switch to
  "provisioned" if you have predictable I/O.

## Pre-existing limitations carried over

- Holo3 occasionally emits clicks instead of scrolls on detail pages — the
  `MicroPlanRunner` already has a scroll-fail-as-success fallback.
- The default residential proxy is geo-targeted to Miami via IPRoyal —
  override `proxy_city` / `proxy_state` in the request body for other
  markets.
