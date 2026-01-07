#!/bin/bash

# Function to perform base64 URL encoding
base64url_encode() {
    echo -n "$1" | base64 | tr '+/' '-_' | tr -d '='
}

# Function to perform base64 URL decoding
base64url_decode() {
    local len=$((${#1} % 4))
    local result="$1"
    if [ $len -eq 2 ]; then result="$1"'=='
    elif [ $len -eq 3 ]; then result="$1"'='
    fi
    echo "$result" | tr '_-' '/+' | base64 -d
}

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [-d] <string>"
    echo "  -d: Decode mode (optional)"
    exit 1
fi

# Check if the first argument is -d for decoding
if [ "$1" = "-d" ]; then
    if [ $# -ne 2 ]; then
        echo "Error: Missing string to decode"
        exit 1
    fi
    base64url_decode "$2"
else
    base64url_encode "$1"
fi