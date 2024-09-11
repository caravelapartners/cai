#!/bin/bash

# Check if the zone letter argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <zone-letter> [machine-number]"
    exit 1
fi

# Extract the zone letter from the first argument
ZONE_LETTER=$1

# Set the machine number to the second argument if provided, or default to 1
MACHINE_NUMBER=${2:-1}

# Define the zone using the zone letterc
ZONE="us-central1-$ZONE_LETTER"

# Define the machine name using the machine number
MACHINE_NAME="gpu-40a100-c1$ZONE_LETTER-$MACHINE_NUMBER"

# Run the gcloud compute instance create command
gcloud compute instances create $MACHINE_NAME \
    --project=compostus \
    --zone=$ZONE \
    --machine-type=a2-highgpu-1g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=compute-service@compostus.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    --min-cpu-platform=Automatic \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --source-machine-image=cai-a100-gpu

# Print a success message
echo "Created machine $MACHINE_NAME in zone $ZONE"
