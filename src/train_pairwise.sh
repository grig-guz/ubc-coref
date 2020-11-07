#!/bin/bash

nohup python -u coref.py --higher_order 0 --use_bert 0 --cuda_id 2 > pairwise_model_fixspan_sents.out&
