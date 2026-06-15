export CUDA_VISIBLE_DEVICES=5
export FLAGS_use_system_allocator=1
for i in {1..10000}; do python engine.py --api_config_file=tester/api_config/api_config_merged_amp.txt --accuracy=True --test_amp=True >> test_pipeline/cpu_accuracy/cpu_accuracy_14.log 2>&1; done