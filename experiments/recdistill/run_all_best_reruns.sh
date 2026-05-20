#!/bin/bash

set -euo pipefail

# Sequential best-rerun launches.
# Comment out any line you do not want to run.
# All commands use --track so each execution writes to a unique results directory.

# DE
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/bprmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/bprmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/bprmf/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/lgcn/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/lgcn/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/lgcn/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/nmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/nmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de/nmf/citeulike/best.yaml --track

# RRD
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/bprmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/bprmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/bprmf/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/lgcn/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/lgcn/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/lgcn/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/nmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/nmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/rrd/nmf/citeulike/best.yaml --track

# # UnKD
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/bprmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/bprmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/bprmf/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/lgcn/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/lgcn/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/lgcn/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/nmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/nmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/unkd/nmf/citeulike/best.yaml --track

# # HTD
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/bprmf/amazon_cd/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/bprmf/bookcrossing/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/bprmf/citeulike/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/lgcn/amazon_cd/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/lgcn/bookcrossing/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/lgcn/citeulike/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/nmf/amazon_cd/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/nmf/bookcrossing/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/htd/nmf/citeulike/best.yaml --track

# # FTD
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/bprmf/amazon_cd/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/bprmf/bookcrossing/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/bprmf/citeulike/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/lgcn/amazon_cd/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/lgcn/bookcrossing/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/lgcn/citeulike/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/nmf/amazon_cd/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/nmf/bookcrossing/best.yaml --track
python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/ftd/nmf/citeulike/best.yaml --track

# # DE_RRD
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/bprmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/bprmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/bprmf/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/lgcn/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/lgcn/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/lgcn/citeulike/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/nmf/amazon_cd/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/nmf/bookcrossing/best.yaml --track
# python3 -u scripts/recdistill/train_student_from_config.py --config config/presets/recdistill/final_rerun/de_rrd/nmf/citeulike/best.yaml --track
