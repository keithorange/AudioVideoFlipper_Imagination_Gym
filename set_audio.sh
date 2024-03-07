#!/bin/bash

# Function to adjust audio settings based on provided parameters
adjust_audio_settings() {
    audio_state=$1
    volume_level=$2
    hold_length=$3

    # Set the volume using osascript (AppleScript command)
    osascript -e "set Volume  $volume_level"

    # Logic for the audio state
    if [ "$audio_state" -eq 1 ]; then
        echo "Audio is playing."
    else
        echo "Audio is paused."
        # Set the volume using osascript (AppleScript command)
        osascript -e "set Volume 0"
    fi

    # Pause for the duration of hold_length before any next commands
    sleep $hold_length

    echo "Volume Level: $volume_level"
    echo "Hold Length: $hold_length seconds"
}

# Get parameters
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --audio_state)
        audio_state="$2"
        shift
        shift
        ;;
    --volume_level)
        volume_level="$2"
        shift
        shift
        ;;
    --hold_length)
        hold_length="$2"
        shift
        shift
        ;;
    *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

# Last lines of the script
adjust_audio_settings $audio_state $volume_level $hold_length
