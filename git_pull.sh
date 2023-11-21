#!/bin/bash

git pull

project_dir=$(pwd)
sub_dirs=("assertpy" "python_config" "python_file" "python_image" "python_video")

for dir in "${sub_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "Updating repository in $dir"
        (cd "$dir" && git pull)
    fi
done

cd "$project_dir" || exit
