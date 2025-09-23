#!/bin/bash

# Check if exactly two arguments (directories) were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory1> <directory2>"
    exit 1
fi

# Assign the input arguments to variables
DIR1="$1"
DIR2="$2"

# Check if the directories exist
if [ ! -d "$DIR1" ] || [ ! -d "$DIR2" ]; then
    echo "Error: Both arguments must be valid directories."
    exit 1
fi

# Loop through all files in the first directory
for file in "$DIR1"/*
do
    # Get just the filename from the full path (e.g., "report.txt")
    filename=$(basename "$file")

    # Construct the full path to the potential matching file in the second directory
    file2="$DIR2/$filename"

    # Check if a file with the same name exists in the second directory
    if [[ -f "$file2" ]]; then

        echo "--- Found matching file: $filename ---"
        
        if diff -q "$file" "$file2" > /dev/null; then
            echo "Files are identical."
        else
            echo "Files differ. Diff output:"
            diff -u "$file" "$file2"
        fi
        
        echo "------------------------------------"
        echo "" # Print a blank line for readability
    fi
done
