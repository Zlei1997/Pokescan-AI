#!/bin/bash

ARCH=./arch.json
COMPILE_KV260=./outputs_kv260

vai_c_pytorch2 \
  --model poke_model_quantized.h5 \
  --arch ${ARCH} \
  --output_dir ${COMPILE_KV260} \
  --net_name pokescan_model
