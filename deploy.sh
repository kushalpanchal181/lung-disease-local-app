#!/bin/bash
set -e

GIT_COMMIT=$(git rev-parse --short HEAD)

echo "GIT_COMMIT=$GIT_COMMIT" > .env

echo "Deploying Git commit: $GIT_COMMIT"

sudo docker compose up -d --build

echo "Deployment complete."
