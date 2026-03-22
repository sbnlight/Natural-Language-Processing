#!/usr/bin/env bash
# run.sh — CS288 HW3 Autograder Entry Point
#
# Autograder invocation method (fixed, cannot be modified):
#   bash run.sh <questions_txt_path> <predictions_out_path>
#
# $1 = Input: path to questions file (one question per line)
# $2 = Output: path to write predicted answers (one answer per line, number of lines must exactly match input)

python3 evaluate_rag_model.py "$1" "$2"