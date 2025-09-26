#!/bin/bash

# Get all global IP addresses
IP_ADDRESSES=$(ip addr show | grep -E "inet .* global" | awk '{print $2}' | cut -d/ -f1)

# Select first non-loopback address
SELECTED_IP=""
for ip in $IP_ADDRESSES; do
    if [[ $ip != "127.0.0.1" ]]; then
        SELECTED_IP=$ip
        break
    fi
done

if [ -z "$SELECTED_IP" ]; then
    echo "Error: Could not find suitable IP address"
    exit 1
fi

echo "Using IP address: $SELECTED_IP to initialize Docker Swarm"

if ! docker info | grep -q "Swarm: active"; then
    docker swarm init --advertise-addr $SELECTED_IP
fi

echo "Pulling Docker image vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204..."
docker pull vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204

# Check if service already exists
if docker service ls | grep -q "code-sandbox"; then
    echo "Service code-sandbox already exists, updating..."
    docker service update \
        --image vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204 \
        --replicas 48 \
        code-sandbox
else
    echo "Creating new service code-sandbox..."
    docker service create \
        --name code-sandbox \
        --replicas 48 \
        --publish 8999:8080 \
        vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204
fi