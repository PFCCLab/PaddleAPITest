export CUDA_VISIBLE_DEVICES=7
export FLAGS_use_system_allocator=1
for i in {1..10000}; do python engine.py --api_config_file=tester/api_config/api_config_merged_getset_item_4.txt --paddle_only=True >> test_pipeline/paddle_only/paddleonly_16.log 2>&1; done