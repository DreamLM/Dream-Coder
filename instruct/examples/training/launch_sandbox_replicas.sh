#!/bin/bash

# Configuration
IMAGE_NAME="vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204"
SERVICE_NAME="code-sandbox"
REPLICAS=48
PUBLISHED_PORT=8999
CONTAINER_PORT=8080
WORK_DIR="/tmp/singularity-swarm"
PID_DIR="$WORK_DIR/pids"
LOG_DIR="$WORK_DIR/logs"
SIF_FILE="$WORK_DIR/code-sandbox.sif"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Create necessary directories
setup_directories() {
    mkdir -p "$WORK_DIR" "$PID_DIR" "$LOG_DIR"
}

# Get all global IP addresses
get_ip_address() {
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
        error "Could not find suitable IP address"
        exit 1
    fi

    echo "$SELECTED_IP"
}

# Convert Docker image to Singularity format
pull_and_convert_image() {
    log "Converting Docker image to Singularity format..."

    # Convert main application image
    if [[ -f "$SIF_FILE" ]] && [[ $(find "$SIF_FILE" -mtime -1 2>/dev/null) ]]; then
        log "Using existing SIF file: $SIF_FILE"
    else
        log "Building Singularity image from Docker registry: $IMAGE_NAME"
        if singularity build --force "$SIF_FILE" "docker://$IMAGE_NAME"; then
            log "Successfully converted image to: $SIF_FILE"
        else
            error "Failed to convert Docker image to Singularity format"
            exit 1
        fi
    fi
}

# Start a single container instance
start_container() {
    local instance_id="$1"
    local port="$2"
    local container_name="${SERVICE_NAME}_${instance_id}"
    local pid_file="$PID_DIR/${container_name}.pid"
    local log_file="$LOG_DIR/${container_name}.log"

    # Check if already running
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        log "Container $container_name already running (PID: $(cat "$pid_file"))"
        return 0
    fi

    log "Starting container $container_name on port $port..."

    # Start container with unique port for each instance
    # Each container gets its own port to avoid conflicts
    nohup singularity run \
        --bind /tmp \
        --env "_BYTEFAAS_RUNTIME_PORT=$port" \
        "$SIF_FILE" > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    # Give container time to start
    sleep 3

    if kill -0 "$pid" 2>/dev/null; then
        # Wait a bit more and check if the port is actually being used
        sleep 2
        if netstat -ln 2>/dev/null | grep -q ":$port "; then
            log "Container $container_name started successfully on port $port (PID: $pid)"
            return 0
        else
            warn "Container $container_name started but may not be listening on port $port yet"
            return 0
        fi
    else
        error "Failed to start container $container_name"
        rm -f "$pid_file"
        return 1
    fi
}

# Stop a container instance
stop_container() {
    local instance_id="$1"
    local container_name="${SERVICE_NAME}_${instance_id}"
    local pid_file="$PID_DIR/${container_name}.pid"
    local base_port=$((PUBLISHED_PORT + 1))
    local port=$((base_port + instance_id - 1))

    log "Stopping container $container_name..."

    # Method 1: Kill by PID file
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Killing process $pid..."
            kill "$pid" 2>/dev/null
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                warn "Force killing process $pid"
                kill -9 "$pid" 2>/dev/null
            fi
        fi
        rm -f "$pid_file"
    fi

    # Method 2: Kill by process pattern matching
    local pids=$(ps aux | grep -E "singularity.*${SIF_FILE##*/}" | grep -v grep | awk '{print $2}')
    if [[ -n "$pids" ]]; then
        log "Killing singularity processes: $pids"
        echo "$pids" | xargs -r kill 2>/dev/null
        sleep 2
        # Force kill if still running
        echo "$pids" | xargs -r kill -9 2>/dev/null
    fi

    # Method 3: Kill processes using the specific port
    if command -v fuser >/dev/null; then
        fuser -k ${port}/tcp 2>/dev/null && log "Freed port $port"
    elif command -v lsof >/dev/null; then
        local port_pids=$(lsof -ti:$port 2>/dev/null)
        if [[ -n "$port_pids" ]]; then
            log "Killing processes using port $port: $port_pids"
            echo "$port_pids" | xargs -r kill -9 2>/dev/null
        fi
    fi

    # Method 4: Kill by command line pattern
    pkill -f "PORT=$port" 2>/dev/null

    log "Container $container_name cleanup completed"
}

# Check container status
check_container_status() {
    local instance_id="$1"
    local container_name="${SERVICE_NAME}_${instance_id}"
    local pid_file="$PID_DIR/${container_name}.pid"

    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Start all container replicas
start_service() {
    log "Starting service $SERVICE_NAME with $REPLICAS replicas..."

    local base_port=$((PUBLISHED_PORT + 1))
    local started=0

    for ((i=1; i<=REPLICAS; i++)); do
        local port=$((base_port + i - 1))
        if start_container "$i" "$port"; then
            ((started++))
        fi
    done

    log "Started $started/$REPLICAS container replicas"
}

# Stop all container replicas
stop_service() {
    log "Stopping service $SERVICE_NAME..."

    # Stop individual containers
    local stopped=0
    for ((i=1; i<=REPLICAS; i++)); do
        stop_container "$i"
        ((stopped++))
    done

    # Additional cleanup: kill any remaining singularity processes related to our service
    log "Performing additional cleanup..."

    # Kill any remaining processes using our SIF file
    local remaining_pids=$(ps aux | grep -E "singularity.*${SIF_FILE##*/}" | grep -v grep | awk '{print $2}')
    if [[ -n "$remaining_pids" ]]; then
        log "Killing remaining singularity processes: $remaining_pids"
        echo "$remaining_pids" | xargs -r kill -9 2>/dev/null
    fi

    # Free up all our port range
    local base_port=$((PUBLISHED_PORT + 1))
    for ((i=0; i<REPLICAS; i++)); do
        local port=$((base_port + i))
        if command -v fuser >/dev/null; then
            fuser -k ${port}/tcp 2>/dev/null
        elif command -v lsof >/dev/null; then
            local port_pids=$(lsof -ti:$port 2>/dev/null)
            if [[ -n "$port_pids" ]]; then
                echo "$port_pids" | xargs -r kill -9 2>/dev/null
            fi
        fi
    done

    # Clean up PID files
    rm -f "$PID_DIR"/${SERVICE_NAME}_*.pid

    # Kill any processes that might be using our work directory
    pkill -f "$WORK_DIR" 2>/dev/null || true

    log "Stopped and cleaned up $stopped container replicas"
    log "Service cleanup completed"
}

# Show service status
show_status() {
    log "Service Status: $SERVICE_NAME"
    echo -e "${BLUE}INSTANCE\t\tSTATUS\t\tPID\t\tPORT\t\tLOG${NC}"
    echo "--------------------------------------------------------------------------------"

    local running=0
    local base_port=$((PUBLISHED_PORT + 1))
    for ((i=1; i<=REPLICAS; i++)); do
        local container_name="${SERVICE_NAME}_${i}"
        local pid_file="$PID_DIR/${container_name}.pid"
        local log_file="$LOG_DIR/${container_name}.log"
        local port=$((base_port + i - 1))

        if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            local pid=$(cat "$pid_file")
            local port_status=""
            if netstat -ln 2>/dev/null | grep -q ":$port "; then
                port_status="✓"
            else
                port_status="✗"
            fi
            printf "%-15s\t%-10s\t%-10s\t%-10s\t%s\n" "$container_name" "Running" "$pid" "$port$port_status" "$log_file"
            ((running++))
        else
            printf "%-15s\t%-10s\t%-10s\t%-10s\t%s\n" "$container_name" "Stopped" "-" "$port" "$log_file"
        fi
    done

    log "Running: $running/$REPLICAS replicas"
    log "Port range: $((base_port)) to $((base_port + REPLICAS - 1))"
}

# Update service (restart with new image)
update_service() {
    log "Updating service $SERVICE_NAME..."

    # Pull new image
    rm -f "$SIF_FILE"  # Force re-download
    pull_and_convert_image

    # Rolling update - restart containers one by one
    local base_port=$((PUBLISHED_PORT + 1))
    for ((i=1; i<=REPLICAS; i++)); do
        if check_container_status "$i"; then
            log "Updating replica $i..."
            stop_container "$i"
            sleep 2
            local port=$((base_port + i - 1))
            start_container "$i" "$port"
            sleep 3  # Brief pause between updates
        fi
    done

    log "Service update completed"
}

# Main script logic
main() {
    setup_directories

    local selected_ip=$(get_ip_address)
    log "Using IP address: $selected_ip for Singularity service"

    # Pull and convert Docker image
    pull_and_convert_image

    # Check if service is already running
    local running_count=0
    for ((i=1; i<=REPLICAS; i++)); do
        if check_container_status "$i"; then
            ((running_count++))
        fi
    done

    if [[ $running_count -gt 0 ]]; then
        log "Service $SERVICE_NAME already has $running_count/$REPLICAS replicas running"
        read -p "Do you want to update the service? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            update_service
        else
            log "Service status:"
            show_status
        fi
    else
        log "Creating new service $SERVICE_NAME..."
        start_service
    fi

    # Show final status
    echo
    show_status

    log "Service deployment complete!"
}

# Handle command line arguments
case "${1:-start}" in
    "start"|"deploy")
        main
        ;;
    "stop")
        stop_service
        ;;
    "status"|"ls")
        show_status
        ;;
    "update")
        update_service
        ;;
    "restart")
        stop_service
        sleep 2
        main
        ;;
    "debug")
        log "Debug information:"
        echo "Work directory: $WORK_DIR"
        echo "SIF file: $SIF_FILE"
        echo "Port range: $((PUBLISHED_PORT + 1)) to $((PUBLISHED_PORT + REPLICAS))"
        echo
        echo "Currently used ports:"
        netstat -ln 2>/dev/null | grep -E ":90[0-9][0-9] " || echo "No ports in 9000 range found"
        echo
        echo "Available tools:"
        echo "socat: $(command -v socat || echo 'not found')"
        echo "nc: $(command -v nc || echo 'not found')"
        echo
        echo "Log files with errors:"
        find "$LOG_DIR" -name "*.log" -exec grep -l "ERROR\|error\|Error" {} \; 2>/dev/null | head -5
        ;;
    "logs")
        local instance="${2:-1}"
        local log_file="$LOG_DIR/${SERVICE_NAME}_${instance}.log"
        if [[ -f "$log_file" ]]; then
            echo "=== Logs for ${SERVICE_NAME}_${instance} ==="
            tail -50 "$log_file"
        else
            error "Log file not found: $log_file"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status|update|restart|debug|logs [instance]}"
        echo "  start/deploy - Start the service (default)"
        echo "  stop         - Stop all containers"
        echo "  status/ls    - Show service status"
        echo "  update       - Update service with new image"
        echo "  restart      - Restart the entire service"
        echo "  debug        - Show debug information"
        echo "  logs [N|lb]  - Show logs for instance N (default: 1)"
        exit 1
        ;;
esac