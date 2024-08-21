#!/usr/bin/env bash

DOCKER_USERNAME="skeidelab"
DOCKER_IMAGE="python-julia-afni"
DOCKER_TAG="0.2"

docker run --rm repronim/neurodocker:1.0.0 generate docker \
    --pkg-manager yum \
    --base-image fedora:36 \
    --afni method=binaries version=latest \
    --env PYTHON_JULIAPKG_PROJECT=/opt/miniconda-latest/julia_env \
    --miniconda \
        version=latest \
        conda_install="matplotlib nilearn pybids pyjuliacall seaborn" \
    --run "python3 -c '\
import juliapkg as jpkg; \
jpkg.add(\"InlineStrings\", uuid=\"842dd82b-1e85-43dc-bf29-5d0ee9dffc48\"); \
jpkg.add(\"MixedModels\", uuid=\"ff71e718-51f3-5ec2-a782-8ffcbfa3c316\"); \
jpkg.add(\"PythonCall\", uuid=\"6099a3de-0909-46bc-b1f4-468b9a2dfc0d\"); \
jpkg.add(\"StatsBase\", uuid=\"2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91\"); \
jpkg.add(\"Suppressor\", uuid=\"fd094767-a336-5f1f-9728-57cf17d0bbfb\"); \
jpkg.add(\"Tables\", uuid=\"bd369af6-aec1-5ad0-b16a-f7cc5008161c\"); \
from juliacall import Main as jl; \
jl.seval(\"using Pkg; Pkg.instantiate(); Pkg.precompile()\")'" > ../Dockerfile

docker build --tag "${DOCKER_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}" ..

docker push "${DOCKER_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}"
