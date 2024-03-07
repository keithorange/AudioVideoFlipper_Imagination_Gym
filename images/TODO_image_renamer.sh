#!/bin/bash

# Path to the directory containing the PNG files
DIRECTORY="."

# Renaming 'example_white.png' to 'example.png' first to avoid overwriting
for file in "$DIRECTORY"/*_white.png; do
    if [[ -f $file ]]; then
        newname="${file/_white.png/.png}"
        mv "$file" "$newname"
    fi
done

# Renaming 'example.png' to 'example_black.png'
for file in "$DIRECTORY"/*.png; do
    if [[ -f $file && ! $file =~ "_white.png$" ]]; then
        newname="${file/.png/_black.png}"
        mv "$file" "$newname"
    fi
done

echo "Renaming complete."

