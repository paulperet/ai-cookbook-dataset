# Kaito: A Kubernetes Operator for Simplified AI Model Deployment

## Introduction

Kaito is a Kubernetes operator that automates AI/ML inference model deployment in Kubernetes clusters. Unlike traditional VM-based deployment approaches, Kaito offers several key advantages:

- **Containerized Model Management**: Packages model files within container images with built-in HTTP servers for inference
- **Preset Configurations**: Eliminates manual GPU hardware tuning with predefined configurations
- **Auto-Provisioning**: Automatically provisions GPU nodes based on model requirements
- **Public Registry Hosting**: Hosts large model images in Microsoft Container Registry (MCR) when licenses permit

This guide walks you through deploying and fine-tuning models using Kaito's streamlined workflow.

## Prerequisites

Before starting, ensure you have:

1. A Kubernetes cluster with GPU nodes available
2. `kubectl` configured to access your cluster
3. Kaito installed in your cluster (see [installation guide](https://github.com/Azure/kaito/blob/main/docs/installation.md))

## Understanding Kaito's Architecture

Kaito uses the Kubernetes Custom Resource Definition (CRD) pattern. You manage a `Workspace` CRD that describes GPU requirements and inference specifications. Kaito controllers automate deployment by reconciling this resource.

Key components include:

- **Workspace Controller**: Creates `Machine` resources for node provisioning and deploys inference workloads
- **Node Provisioner Controller**: Integrates with AKS APIs to add GPU nodes using Karpenter-core APIs

## Step 1: Create a Fine-Tuning Workspace

Create a YAML file for your fine-tuning workspace. This example fine-tunes the Phi-3-mini model using QLoRA:

```yaml
# workspace-tuning-phi-3.yaml
apiVersion: kaito.sh/v1alpha1
kind: Workspace
metadata:
  name: workspace-tuning-phi-3
resource:
  instanceType: "Standard_NC6s_v3"
  labelSelector:
    matchLabels:
      app: tuning-phi-3
tuning:
  preset:
    name: phi-3-mini-128k-instruct
  method: qlora
  input:
    urls:
      - "https://huggingface.co/datasets/philschmid/dolly-15k-oai-style/resolve/main/data/train-00000-of-00001-54e3756291ca09c6.parquet?download=true"
  output:
    image: "ACR_REPO_HERE.azurecr.io/IMAGE_NAME_HERE:0.0.1"
    imagePushSecret: ACR_REGISTRY_SECRET_HERE
```

**Configuration Breakdown:**

- `instanceType`: Specifies the GPU instance type (Standard_NC6s_v3 in this example)
- `preset.name`: Defines the base model to fine-tune (phi-3-mini-128k-instruct)
- `method`: Specifies the fine-tuning technique (qlora for QLoRA)
- `input.urls`: Points to the training dataset
- `output.image`: Target Azure Container Registry path for the fine-tuned model
- `imagePushSecret`: Kubernetes secret for ACR authentication

## Step 2: Apply the Workspace Configuration

Deploy the workspace to your Kubernetes cluster:

```bash
kubectl apply -f workspace-tuning-phi-3.yaml
```

## Step 3: Monitor Workspace Status

Track the deployment progress by checking the workspace status:

```bash
kubectl get workspace workspace-tuning-phi-3
```

The command displays several readiness columns:
- `RESOURCEREADY`: Indicates if GPU resources are provisioned
- `INFERENCEREADY`: Shows if the inference service is running
- `WORKSPACEREADY`: Confirms the entire workspace is operational

Wait until `WORKSPACEREADY` shows `True`, indicating successful deployment.

## Step 4: Access the Inference Service

Once deployed, retrieve the service details:

```bash
kubectl get svc workspace-tuning-phi-3
```

Note the `CLUSTER-IP` address for the service. This internal IP hosts the inference endpoint.

## Step 5: Test the Inference Endpoint

Create a temporary curl pod to test the fine-tuned model:

```bash
# Export the cluster IP for easier access
export CLUSTERIP=$(kubectl get svc workspace-tuning-phi-3 -o jsonpath="{.spec.clusterIPs[0]}")

# Send a test query to the inference endpoint
kubectl run -it --rm --restart=Never curl \
  --image=curlimages/curl \
  -- curl -X POST http://$CLUSTERIP/chat \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Explain quantum computing in simple terms\"}"
```

The service responds with the model's generated completion based on your prompt.

## Key Takeaways

1. **Simplified Workflow**: Kaito abstracts complex GPU provisioning and model deployment tasks
2. **Preset Configurations**: Eliminates manual tuning for common model architectures
3. **Kubernetes Native**: Integrates seamlessly with existing Kubernetes tooling and practices
4. **Fine-Tuning Support**: Includes built-in support for parameter-efficient fine-tuning methods like QLoRA

## Next Steps

- Explore additional [preset models](https://github.com/Azure/kaito/tree/main/presets) supported by Kaito
- Configure external access via Ingress or LoadBalancer services
- Set up monitoring and logging for your inference workloads
- Implement CI/CD pipelines for automated model updates

For advanced configurations and troubleshooting, refer to the [Kaito documentation](https://github.com/Azure/kaito).