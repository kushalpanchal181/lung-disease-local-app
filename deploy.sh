#!/bin/bash
set -e

GIT_COMMIT=$(git rev-parse --short HEAD)

GIT_COMMIT=$(git rev-parse --short HEAD)
DEPLOYMENT_DATE=$(date +%Y-%m-%d)

echo "GIT_COMMIT=$GIT_COMMIT" > .env
echo "DEPLOYMENT_DATE=$DEPLOYMENT_DATE" >> .env

sudo docker compose up -d --build

echo "Deployment complete."
