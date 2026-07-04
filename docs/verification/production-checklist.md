# Production Checklist

## Overview
Before deploying the ARMT-GAN Platform to a live Web production environment, ensure all items in this checklist are complete.

## Security
- [ ] Database credentials and secrets are managed via a secure vault (e.g., HashiCorp Vault, AWS Secrets Manager).
- [ ] TLS 1.3 is enforced on all public Ingress controllers.
- [ ] Docker images have zero known critical vulnerabilities.
- [ ] API Rate limiting is configured.

## Performance
- [ ] Redis caching is enabled and tested.
- [ ] PostgreSQL connection pooling (e.g., PgBouncer) is active.
- [ ] Frontend static assets are served via a CDN.
- [ ] Kubernetes Horizontal Pod Autoscaler (HPA) is configured for both web API and AI worker nodes.

## Observability
- [ ] Centralized logging is configured (e.g., ELK, Datadog).
- [ ] Application metrics (Prometheus format) are exposed by FastAPI and Next.js.
- [ ] Alerts are configured for high error rates or sustained high queue depths.

## Data
- [ ] Automated database backups are configured and tested.
- [ ] Object Storage bucket policies restrict public access appropriately.
