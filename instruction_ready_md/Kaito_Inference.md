# Kaito Inference Guide: Deploying and Querying Models on Kubernetes

## Overview

Kaito is a Kubernetes operator that simplifies AI/ML inference model deployment by automating GPU provisioning, container image management, and inference service setup. This guide walks you through deploying and querying models using Kaito's Workspace custom resources.

## Prerequisites

Before starting, ensure you have:
- A Kubernetes cluster with GPU nodes available
- Kaito installed in your cluster (follow the [installation guide](https://github.com/Azure/kaito/blob/main/docs/installation.md))
- `kubectl` configured to access your cluster

## Part 1: Deploying Phi-3 Mini for Inference

### Step 1: Create the Workspace Configuration

Create a file named `phi-3-inference.yaml` with the following content:

```yaml
apiVersion: kaito.sh/v1alpha1
kind: Workspace
metadata:
  name: workspace-phi-3-mini
resource:
  instanceType: "Standard_NC6s_v3"
  labelSelector:
    matchLabels:
      app: phi-3-adapter
tuning:
  preset:
    name: phi-3-mini-4k-instruct
  method: qlora
  input:
    urls:
      - "https://huggingface.co/datasets/philschmid/dolly-15k-oai-style/resolve/main/data/train-00000-of-00001-54e3756291ca09c6.parquet?download=true"
  output:
    image: "ACR_REPO_HERE.azurecr.io/IMAGE_NAME_HERE:0.0.1"
    imagePushSecret: ACR_REGISTRY_SECRET_HERE
```

**Note:** Replace `ACR_REPO_HERE`, `IMAGE_NAME_HERE`, and `ACR_REGISTRY_SECRET_HERE` with your Azure Container Registry details.

### Step 2: Apply the Workspace Configuration

Deploy the workspace to your Kubernetes cluster:

```bash
kubectl apply -f phi-3-inference.yaml
```

### Step 3: Monitor Deployment Status

Check the workspace status until all components are ready:

```bash
kubectl get workspace workspace-phi-3-mini
```

Wait until the `WORKSPACEREADY` column shows `True`, indicating successful deployment. This process may take several minutes as Kaito provisions GPU resources and deploys the model.

### Step 4: Test the Inference Endpoint

First, retrieve the service's cluster IP:

```bash
kubectl get svc workspace-phi-3-mini-adapter
```

Then, use a temporary curl pod to test the inference endpoint:

```bash
export CLUSTERIP=$(kubectl get svc workspace-phi-3-mini-adapter -o jsonpath="{.spec.clusterIPs[0]}")
kubectl run -it --rm --restart=Never curl --image=curlimages/curl -- curl -X POST http://$CLUSTERIP/chat -H "accept: application/json" -H "Content-Type: application/json" -d "{\"prompt\":\"What is machine learning?\"}"
```

The response will contain the model's answer to your prompt.

## Part 2: Deploying Phi-3 with Custom Adapters

### Step 1: Create the Adapter-Enabled Workspace Configuration

Create a file named `phi-3-with-adapters.yaml` with the following content:

```yaml
apiVersion: kaito.sh/v1alpha1
kind: Workspace
metadata:
  name: workspace-phi-3-mini-adapter
resource:
  instanceType: "Standard_NC6s_v3"
  labelSelector:
    matchLabels:
      app: phi-3-adapter
inference:
  preset:
    name: phi-3-mini-128k-instruct
  adapters:
    - source:
        name: "phi-3-adapter"
        image: "ACR_REPO_HERE.azurecr.io/ADAPTER_HERE:0.0.1"
      strength: "1.0"
```

**Note:** This configuration uses a pre-tuned adapter from your container registry. Ensure you have the adapter image available.

### Step 2: Apply the Adapter Configuration

Deploy the adapter-enabled workspace:

```bash
kubectl apply -f phi-3-with-adapters.yaml
```

### Step 3: Verify Deployment

Monitor the workspace status:

```bash
kubectl get workspace workspace-phi-3-mini-adapter
```

Again, wait for the `WORKSPACEREADY` column to show `True`.

### Step 4: Query the Adapter-Enhanced Model

Retrieve the service endpoint and test inference:

```bash
export CLUSTERIP=$(kubectl get svc workspace-phi-3-mini-adapter -o jsonpath="{.spec.clusterIPs[0]}")
kubectl run -it --rm --restart=Never curl --image=curlimages/curl -- curl -X POST http://$CLUSTERIP/chat -H "accept: application/json" -H "Content-Type: application/json" -d "{\"prompt\":\"Explain how fine-tuning works with adapters.\"}"
```

## Key Concepts

### Workspace Controller
The workspace controller reconciles your Workspace custom resources, handling GPU node provisioning and inference workload deployment based on model presets.

### Node Provisioning
Kaito uses the gpu-provisioner controller (compatible with Karpenter-core APIs) to automatically add GPU nodes to your AKS cluster based on model requirements.

### Model Presets
Kaito provides preset configurations for popular models like Phi-3, eliminating the need for manual deployment parameter tuning.

## Troubleshooting

If your workspace doesn't become ready:
1. Check GPU availability in your cluster: `kubectl get nodes`
2. Verify Kaito controller logs: `kubectl logs -n kaito-system -l app=kaito-controller`
3. Ensure your container registry credentials are correctly configured

## Next Steps

- Explore additional model presets in the [Kaito documentation](https://github.com/Azure/kaito)
- Learn about advanced configuration options for production deployments
- Consider implementing health checks and monitoring for your inference services

By following this guide, you've successfully deployed AI inference models on Kubernetes using Kaito, demonstrating both basic inference and adapter-enhanced deployments.