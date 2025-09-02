#!/bin/bash

# Script để tạo PNG từ tất cả file markdown có Mermaid diagrams

echo "Creating PNG files from Mermaid diagrams..."

# Danh sách các file cần tạo PNG
files=(
    "docs/10-competency.md:assets/competency-matrix.png"
    "docs/deep-theory.md:assets/deep-theory-architecture.png"
    "docs/05-deep-learning.md:assets/deep-learning-architecture.png"
    "docs/14-benchmarks.md:assets/benchmarks-architecture.png"
    "docs/13-interop.md:assets/interop-patterns.png"
    "docs/12-stack.md:assets/technology-stacks.png"
    "docs/02-data-analyst.md:assets/data-analyst-overview.png"
    "docs/09-12-week.md:assets/12-week-roadmap.png"
    "docs/11-tooling.md:assets/development-environment.png"
    "docs/08-projects.md:assets/portfolio-projects.png"
    "docs/15-pytorch.md:assets/pytorch-curriculum.png"
    "docs/01-foundations.md:assets/foundations-overview.png"
    "docs/07-mlops.md:assets/mlops-overview.png"
    "docs/04-time-series.md:assets/time-series-analysis.png"
)

for file_pair in "${files[@]}"; do
    IFS=':' read -r input_file output_file <<< "$file_pair"
    
    echo "Processing $input_file..."
    
    # Tạo PNG
    mmdc -i "$input_file" -o "$output_file" -b transparent
    
    # Tìm và đổi tên file được tạo
    base_name=$(basename "$output_file" .png)
    generated_file=$(find . -name "${base_name}-1.png" -type f 2>/dev/null)
    
    if [ -n "$generated_file" ]; then
        mv "$generated_file" "$output_file"
        echo "✅ Created $output_file"
    else
        echo "❌ Failed to create $output_file"
    fi
done

echo "All PNG files created!"