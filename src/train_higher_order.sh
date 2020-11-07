#!/bin/bash
nohup python -u coref.py --higher_order 1 --use_bert 0 --cuda_id 1 > higher_order_model.out&
