export CUDA_VISIBLE_DEVICES=3
export FLAGS_use_system_allocator=1
# export PYTHONPATH=/host_home/wanghuan29/Paddle/build/python
# sleep 3h
for i in {1..10000}; do python engine.py --api_config_file=test_pipeline/gpu_0size/gpu_0size_accuracy/0size_tensor_accuracy_10.txt --accuracy=True >> test_pipeline/gpu_0size/gpu_0size_accuracy/0size_tensor_accuracy_10.log 2>&1; done
