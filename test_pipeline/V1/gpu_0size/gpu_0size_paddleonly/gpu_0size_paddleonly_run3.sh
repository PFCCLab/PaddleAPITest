export CUDA_VISIBLE_DEVICES=0
export FLAGS_use_system_allocator=1
# export PYTHONPATH=/host_home/wanghuan29/Paddle/build/python
# sleep 3h
for i in {1..10000}; do python engine.py --api_config_file=test_pipeline/gpu_0size/gpu_0size_paddleonly/0sizetensor_paddleonly_3.txt --paddle_only=True >> test_pipeline/gpu_0size/gpu_0size_paddleonly/0sizetensor_paddleonly_3.log 2>&1; done
