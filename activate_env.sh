#!/bin/bash
# Source conda.sh to ensure conda command is available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the desired conda environment
conda activate slide-synth-studio

# Start a new shell session in the activated environment
PS1="(slide-synth-studio) \u@\h:\w# " bash --noprofile --norc