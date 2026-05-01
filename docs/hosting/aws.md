# AWS (EKS)

Self-hosted on EKS using one GPU node group, ECR, Secrets Manager, EFS, and an ALB ingress. The full Terraform + k8s manifests + runbook are at [`deploy/aws/`](https://github.com/mercurialsolo/mantis/tree/main/deploy/aws).

## Architecture

```
ALB ingress (HTTPS, ACM cert)
   ↓
EKS cluster
  • GPU NodeGroup (g6e.2xlarge L40S, on-demand)
  • CPU NodeGroup (m6i, for ALB ctrl + system add-ons)
   ↓
EFS volume (mantis-data — runs, checkpoints, profiles, recordings)
ECR (mantis-prod-server image)
Secrets Manager (anthropic_api_key, proxy_*, mantis_api_token)
External Secrets Operator → in-pod env
```

## Footprint

| Resource | Type | Cost (us-east-1) |
|---|---|---|
| GPU node | `g6e.2xlarge` (L40S 48 GB) | ~$1.86/hr on-demand |
| EFS Standard | per-GB-month | $0.30/GB-month |
| ALB | Application Load Balancer | ~$22/month + LCU |
| ECR | image registry | $0.10/GB-month after free tier |

For Holo3 Q8_0 (~34 GB VRAM) the L40S is the cheapest fit. If you need more headroom, jump to `p4d.24xlarge` (8× A100, reserved-only, expensive).

## End-to-end deploy

The detailed runbook is in [`deploy/aws/README.md`](https://github.com/mercurialsolo/mantis/blob/main/deploy/aws/README.md). High-level:

1. **Build + push image to ECR**
   ```bash
   docker build --platform linux/amd64 -f docker/server.Dockerfile \
     -t "$ECR/$REPO:$(git rev-parse --short HEAD)" .
   docker push "$ECR/$REPO:$(git rev-parse --short HEAD)"
   ```

2. **Provision infra**
   ```bash
   cd deploy/aws/terraform
   terraform init
   terraform apply -var "cluster_name=mantis-prod" \
                   -var "image_tag=$(git rev-parse --short HEAD)" \
                   -var "vpc_id=vpc-..." \
                   -var "private_subnet_ids=[subnet-..., subnet-...]" \
                   -var "node_security_group_id=sg-..."
   ```

3. **Populate secrets in Secrets Manager**
   ```bash
   for k in anthropic_api_key proxy_url proxy_user proxy_pass mantis_api_token; do
     aws secretsmanager put-secret-value \
       --secret-id "mantis-prod/$k" \
       --secret-string "$(read -rp "$k: " v && echo "$v")"
   done
   ```

4. **Pre-warm Holo3 weights** (one-time HF download to EFS PVC)
   ```bash
   kubectl apply -f deploy/aws/k8s/init-holo3-weights.yaml
   kubectl logs -f job/init-holo3-weights
   ```

5. **Deploy the workload**
   ```bash
   kubectl apply -f deploy/aws/k8s/external-secret.yaml
   kubectl apply -f deploy/aws/k8s/deployment.yaml
   kubectl apply -f deploy/aws/k8s/service.yaml
   kubectl apply -f deploy/aws/k8s/ingress.yaml
   ```

6. **Smoke-test**
   ```bash
   HOST=$(kubectl get ingress mantis-holo3-server \
     -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
   TOK=$(aws secretsmanager get-secret-value \
     --secret-id mantis-prod/mantis_api_token \
     --query SecretString --output text)

   curl -fsS -X POST "https://$HOST/v1/predict" \
     -H "X-Mantis-Token: $TOK" \
     -H "Content-Type: application/json" \
     -d '{"detached": true, "micro": "plans/example/extract_listings.json", "state_key": "smoke", "max_cost": 2}'
   ```

## Operational notes

- **Scale to zero:** the deployment defaults to `replicas: 1` (always-on GPU). For bursty traffic, drop to `min_replicas: 0` and use [KEDA](https://keda.sh/) or [Karpenter](https://karpenter.sh/) — without scale-to-zero you're paying ~$45/day per replica.
- **EFS throughput:** defaults to `throughput_mode = elastic` (pay-per-use). Switch to `provisioned` for predictable I/O.
- **Image rollouts:** `kubectl set image deployment/mantis-holo3-server server=<new-image>:<tag>` rolls forward; tag images with the git SHA for idempotent deploys.
- **Tenant keys hot reload:** mount `mantis_tenant_keys` as a Secret with [`reloader`](https://github.com/stakater/Reloader) annotations, OR use the built-in 5-second cache (no pod restart needed for token rotation as long as the secret is updated in Secrets Manager).

## Status

The Terraform + k8s manifests are starter scaffolding — they assume an existing EKS cluster (this PR doesn't create the control plane). Review `deploy/aws/terraform/main.tf` for your VPC / IAM constraints before `terraform apply`.

## See also

- [`deploy/aws/README.md`](https://github.com/mercurialsolo/mantis/blob/main/deploy/aws/README.md) — full runbook
- [`deploy/aws/k8s/`](https://github.com/mercurialsolo/mantis/tree/main/deploy/aws/k8s) — manifests
- [Tenant keys](../operations/tenant-keys.md) — how to set up the multi-tenant keys file
- [Metrics](../operations/metrics.md) — Prometheus scrape via the ALB
