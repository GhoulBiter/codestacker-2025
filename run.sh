# Detect OS and set the appropriate hostname for host access
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # On Linux, Docker doesn’t automatically resolve host.docker.internal,
    # so we use the special host gateway mapping.
    export DOCKER_HOST_IP="host.docker.internal"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS automatically supports host.docker.internal
    export DOCKER_HOST_IP="host.docker.internal"
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    export DOCKER_HOST_IP="host.docker.internal"
else
    # Default fallback
    export DOCKER_HOST_IP="host.docker.internal"
fi

export MLFLOW_TRACKING_URI="http://${DOCKER_HOST_IP}:5000"

docker-compose up --build
