#!/bin/bash

project_dir=$(pwd)
sub_dirs=("assert_utils" "python_utils")

for dir in "${sub_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "Updating repository in $dir"
        (cd "$dir" && git pull)
    fi
done

cd "$project_dir" || exit
