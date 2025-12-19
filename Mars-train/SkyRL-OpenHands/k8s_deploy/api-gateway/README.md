# API Gateway

This shows a simple API gateway that routes requests to the appropriate replica based on image ID. All the routing logic is implemented in [`main.go`](main.go).

# Deployment Instructions

First, make sure you are in the right directory:
```bash
cd k8s_deploy/api-gateway
```
## (Optional) Build the image

You can build the image with the following command:

```bash
docker build -t api-gateway .
docker push username/api-gateway:latest
```

The below commands use the pre-built image `sumanthrh/oh-api-gateway:latest`


## Deployment with AWS EKS

The API gateway K8s yaml is provided in `api-gateway/deploy.yaml`.

### Deploy node group

For example, you can use the following command to deploy a node group with 2 m6i.xlarge nodes:
```bash
eksctl create nodegroup --name api-gateway --node-type m6i.xlarge  --nodes 2 --region us-west-2 --cluster openhands-cluster
```

This should be plenty for the API gateway.

### Deploy API Gateway and K8s service

Deploy the API gateway pods with the following command:

```bash
kubectl apply -f deploy.yaml
```

You should see the API gateway pods running in the cluster.

```bash
kubectl get pods
```
```
NAME                          READY   STATUS    RESTARTS   AGE
api-gateway-d4d7c6995-8qwl9   1/1     Running   0          5s
api-gateway-d4d7c6995-h7lcg   1/1     Running   0          5s
```

You should also see the K8s service created:

```bash
kubectl get svc
```

```
NAME              TYPE           CLUSTER-IP       EXTERNAL-IP                                                              PORT(S)          AGE
api-gateway-svc   LoadBalancer   10.xx.xx.xx  mycustomstring.us-west-2.elb.amazonaws.com   3000:32190/TCP   5s
```
