# OpenHands Remote Runtime

This directory contains the implementation of the remote runtime server for OpenHands.

For scaling to multiple replicas, we have also made lightweight client-side modifications in [openhands/runtime/impl/remote/remote_runtime.py](../impl/remote/remote_runtime.py).


## Quick Start

Launching a single server is simple:

### 0. Start docker daemon

On Linux, you can run:
```bash
sudo dockerd
```

On MacOS, you should install Docker Desktop and start the application.

### 1. Install dependencies

You can install the dependencies with poetry:

```bash
poetry install
```

### 2. Launch the server

```bash
ALLHANDS_API_KEY=<your_api_key> python -m openhands.runtime.remote_runtime_server.main --host ${HOST} --port ${PORT}
```

Make sure to set `HOST` to be the _public_ IP for your machine! `ALLHANDS_API_KEY` can be any key of your choice - this is used for authentication during evaluation. 

### 3. Run evaluations

First, make sure to configure your LLM in `config.toml` file, following instructions from [OpenHands](https://github.com/all-hands-ai/openhands). Then, run the following command:

```bash
ALLHANDS_API_KEY=<your_api_key> \
RUNTIME=remote \
SANDBOX_REMOTE_RUNTIME_API_URL="http://${HOST}:${PORT}" \
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.gpt_4o_mini HEAD CodeActAgent 1 10 1 "princeton-nlp/SWE-bench_Lite" test
```

## Configuring regular docker cleanup
For running your server for a long time period, it's important to periodically clean up older containers (in case of a network failure or sudden exit during evaluation) as well as images in case your storage space is small. We provide basic scripts for cleaning containers and images to be used as cron jobs. Our pre-built [image](https://hub.docker.com/r/sumanthrh/openhands-dev) for Linux comes with all these configured. If you're running on your machine and want to set this up, you can refer to the clean up scripts in [runtime](/containers/runtime/). 

## Image Caching

For usage with training, we highly recommend running image caching to speed up startup time during training. This means that we want to send a build request to the server for each instance we wish to use during training.

We provide a simple script `run_builds.sh`

```bash
ALLHANDS_API_KEY=<your_api_key> \
RUNTIME=remote \
SANDBOX_REMOTE_RUNTIME_API_URL="http://${HOST}:${PORT}" \
python ./evaluation/benchmarks/swe_bench/scripts/run_builds.sh llm.gpt_4o_mini HEAD CodeActAgent 5000 1 64 "princeton-nlp/SWE-bench_Lite" test
```
The format is same as `run_infer.sh`. We ignore `max-iterations` (1 above) and simply trigger a runtime build for each instance.

## Multiple Replicas

Ideally, the client should not be modified to be able to support multiple replicas. However, our implementation of image-ID based routing (in order to be able to spread docker image cache across multiple replicas) is not compatible with the current remote runtime implementation.


Thus we design it to have the following flow:

![image](/assets/client_flow.png)


To be more specific, OpenHands' `RemoteRuntime` client first makes a `/image_exists` request for the current instance's image. Our API gateway routes `/image_exists` requests to the server with the image. In the response, the replica will also return the service URL running on the replica, so that the client can use direct node addressing in future requests to the same replica.


# Usage with OpenHands repository

For using our runtime implementation but with the OpenHands repository, you can simply do the following:

1. Host the Remote Runtime Server

You can host the remote runtime server on a local machine or remotely with Kubernetes following our deployment instructions in [k8s_deploy/README.md](../../../k8s_deploy/README.md).

2. Clone the OpenHands repository

```bash
git clone https://github.com/all-hands-ai/openhands <path_to_openhands_repo>
```

3. Replace the remote runtime client with our implementation:

```bash
cp <path_to_our_repo>/openhands/runtime/impl/remote/remote_runtime.py <path_to_openhands_repo>/openhands/runtime/impl/remote/remote_runtime.py
```

4. Use the right runtime environment variable:

```bash
RUNTIME=remote
ALLHANDS_API_KEY=<your_api_key>
```
