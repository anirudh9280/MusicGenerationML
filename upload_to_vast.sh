#!/bin/bash

# Upload Task 2 Data from Local Machine to Vast.ai
# Run this script from your LOCAL machine after getting Vast.ai instance

echo "=== Upload MAESTRO Data to Vast.ai ==="

# Replace with your actual Vast.ai instance IP
VAST_IP="YOUR_VAST_IP_HERE"

echo "Make sure to replace YOUR_VAST_IP_HERE with your actual Vast.ai instance IP"
echo "You can find this in your Vast.ai dashboard"
echo ""

# Upload the MAESTRO TFRecord data
echo "Uploading MAESTRO TFRecords (~16GB)..."
echo "This may take a while depending on your upload speed..."

scp -r data/maestro_tfrecords/ root@${VAST_IP}:~/MusicGenerationML/data/

echo "Upload complete!"
echo "Now run the vast_ai_setup_commands.sh script on your Vast.ai instance" 