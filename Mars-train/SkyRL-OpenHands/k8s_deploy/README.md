# K8s Deployment on AWS EKS

This section contains instructions for deploying the OpenHands system on a Kubernetes cluster on AWS EKS. Our deployment is meant to be straightforward and as an easy starting point for deploying your own remote OpenHands server.

## Prerequisites

Make sure to setup the AWS and EKS CLI as per the official instructions.

### Create a new EKS cluster

Create a new EKS cluster `openhands-cluster` via the AWS Console or the CLI following the [official instructions](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html).

### Create a security group

Our remote server spawns docker containers and assigns them to a random free port on the server. You would thus need to create a security group that will allow ingress for the ports used by the remote server. Depending on the port you use, this can be different. For the default settings, you would need to allow ingress for the ports in the range 3000-60000.

### Create a node group

We've provided a node group manifest in [`node_group.yaml`](node_group.yaml).

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: openhands-cluster
  # change as needed
  region: us-west-2

nodeGroups:
  - name: nvme-storage-2
    instanceType: i4i.4xlarge
    desiredCapacity: 8
    privateNetworking: false
    volumeSize: 50
    labels:
      node-group: nvme-storage
    securityGroups:
      # Change this to your security group ID
      attachIDs: [sg-<security-group-id>]
    preBootstrapCommands:
      - apt install -y xfsprogs
      # Mount the NVME volume
      - if ! grep -qs '/mnt/nvme' /proc/mounts; then mkfs.xfs -f /dev/nvme1n1; mkdir -p /mnt/nvme; mount /dev/nvme1n1 /mnt/nvme; echo "/dev/nvme1n1 /mnt/nvme xfs defaults 0 0" >> /etc/fstab; fi
```

Note that we use storage optimized i4i instances with large NVMe root volumes since we make heavy use of customized docker images. The `preBootstrapCommands` section is used to mount the NVMe volume to `/mnt/nvme`. Make sure to change the security group ID to the one you used during cluster creation. You can adjust the number of replicas or the instance type as needed.

NOTE: If you wish to change the number of replicas, you will need to change the number of nodes in the node group, the number of pods in your manifest, and the `REPLICA_COUNT` environment variable in the [api-gateway/deploy.yaml](./api-gateway/deploy.yaml) file.

You can then deploy the node group:

```bash
eksctl create nodegroup --config-file node_group.yaml
```

## Deploying the StatefulSet

We use a statefulset to deploy the OpenHands servers. You can view the manifest in [`statefulset_deploy.yaml`](statefulset_deploy.yaml). We use an EFS volume to persist server logs. Before deployment, make sure to create an EFS volume and use the file system ID in the manifest. Further, all communication to the remote server requires the correct API key. Our authentication is extremely basic, and simply uses a fixed API key that can be passed to the server via an environment variable. You can customize this by changing `OPENHANDS_API_KEY` in the manifest.

Then, you can deploy the statefulset with the following command:

```bash
kubectl apply -f statefulset_deploy.yaml
```

You should see the statefulset pods running in the cluster.

```bash
kubectl get pods
```

```
NAME                          READY   STATUS    RESTARTS   AGE
openhands-0                  1/1     Running   0          5s
openhands-1                  1/1     Running   0          5s
openhands-2                  1/1     Running   0          5s
openhands-3                  1/1     Running   0          5s
openhands-4                  1/1     Running   0          5s
openhands-5                  1/1     Running   0          5s
openhands-6                  1/1     Running   0          5s
openhands-7                  1/1     Running   0          5s
```

The entrypoint script should automatically start the OpenHands remote server in a tmux session. You can inspect the logs by running:

```bash
kubectl logs openhands-0
```



## Deploying the API Gateway and the K8s Service

We use an API gateway to route requests to the server replicas. Each replica in itself doesn't have even NVMe storage for all images. Since we wish to spread the load across the server replicas, we route by Image ID in the API gateway. We further deploy a load balancer service for the api gateway pods.

For deployment instructions, see [api-gateway/README.md](./api-gateway/README.md).

## Running Image caching

As with a single standalone server, it is recommended to run image caching to speed up startup time during training.

You can refer to the instructions in the [openhands/runtime/remote_runtime_server/README.md](../openhands/runtime/remote_runtime_server/README.md) file.
