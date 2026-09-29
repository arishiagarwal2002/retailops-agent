# Deployment

## Stage 1: Render + Neon (free tier, about an hour)

1. **Database.** Create a Neon project (Postgres 16) and enable pgvector: `CREATE EXTENSION vector;` in the SQL editor. Copy the connection string, change the scheme to `postgresql+psycopg://` and add `?sslmode=require`. That's `DATABASE_URL`. For `DATABASE_URL_RO`, use the same host with user `retailops_ro` and a strong password of your choice. `scripts/create_ro_role.py` creates that role with exactly that password.
2. **Render.** Push the repo to GitHub, then "New → Blueprint" and select the repo; `render.yaml` defines the API and the UI. Set `DATABASE_URL`, `DATABASE_URL_RO` and `GEMINI_API_KEY` (free) in the dashboard. `render.yaml` sets `LLM_PROVIDER=gemini`.
3. The API's `preDeployCommand` runs `scripts/bootstrap.sh`: migrations, the read-only role, loading the 1M rows from `data/processed/*.parquet`, and policy ingestion. The first deploy takes a few minutes.
4. Create an API key for the demo: Render shell → `python -m scripts.create_api_key public-demo`. Put it in the UI service as `DEMO_API_KEY`, or share it only with reviewers.
5. **Bill protection** (do all four): `DAILY_SPEND_CAP_GBP=1.0`, the default model `claude-haiku-4-5`, an API key gate, and a billing alert in the Anthropic console.

Free-tier caveat: Render's free instances sleep, so the first request after idle takes ~30-60 s. Record the demo video as a fallback.

## Stage 2: AWS App Runner + RDS (the cloud evidence)

```bash
# 1. Image -> ECR
aws ecr create-repository --repository-name retailops-api
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker build -t retailops-api . && docker tag retailops-api <acct>.dkr.ecr.<region>.amazonaws.com/retailops-api:latest
docker push <acct>.dkr.ecr.<region>.amazonaws.com/retailops-api:latest

# 2. Secrets -> SSM Parameter Store (SecureString)
aws ssm put-parameter --name /retailops/DATABASE_URL    --type SecureString --value 'postgresql+psycopg://...'
aws ssm put-parameter --name /retailops/DATABASE_URL_RO --type SecureString --value 'postgresql+psycopg://retailops_ro:...'
aws ssm put-parameter --name /retailops/ANTHROPIC_API_KEY --type SecureString --value 'sk-ant-...'
```

3. **RDS**: PostgreSQL 16, `db.t4g.micro`, in private subnets. pgvector is supported on RDS: run `CREATE EXTENSION vector;` once as the master user.
4. **App Runner**: source = the ECR image, port 8000, health check `/health`. Add a VPC connector to reach RDS. Map the three SSM parameters to environment variables via "Runtime environment secrets". The instance role needs `ssm:GetParameters` and `kms:Decrypt` on those parameters only.
5. Run the bootstrap once as a one-off task (ECS run-task with the same image and `sh scripts/bootstrap.sh`), or from a bastion.
6. Cost guard: an AWS Budgets alert at $10/month, plus the in-code daily LLM cap.

Only list AWS on a CV once this stage is live and you can describe the VPC connector, IAM role and secret wiring from memory.
