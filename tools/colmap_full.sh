# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved

COLMAP_RESULTS_DIR=$1
DATASET_ROOT=$2


# for i in `seq -f '%05g' $1 $2`; do
bash tools/triangulate_colmap.sh $COLMAP_RESULTS_DIR $DATASET_ROOT/train
    # python gaussian-splatting/train.py -s $COLMAP_RESULTS_DIR/$i -i $DATASET_ROOT/train/rgbs -w -m $OUTPUT_DIR/$i
# done
