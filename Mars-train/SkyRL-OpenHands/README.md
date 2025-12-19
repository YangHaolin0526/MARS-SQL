# SkyRL-OpenHands: OpenHands With A Scalable Remote Runtime Server

This repository is a fork of [OpenHands](https://github.com/All-Hands-AI/OpenHands), with the remote runtime server implementation used in [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0). We provide:
1. An efficient remote server implementation for OpenHands: We leverage [aidocker](https://aiodocker.readthedocs.io/en/latest/) and use the [crun](https://github.com/containers/crun) container runtime, which offers lightweight and high-performance container execution. Our setup supports easily running 80–100 containers per replica on nodes with just 16 CPUs.
2. Deployment files for a scalable deployment on Kubernetes: We leverage storage-optimized instances to cache container images and enable fast startup times. Further, we use a simple API gateway to spread the image cache across multiple replicas. 

While there remains significant room for optimization, our current implementation is simple, effective, and publicly available.


## Remote Runtime Server

The remote runtime server implementation is provided in [openhands/runtime/remote_runtime_server](./openhands/runtime/remote_runtime_server/README.md).


## Deployment

For deployment instructions, please refer to the [k8s_deploy](./k8s_deploy/README.md) directory.
