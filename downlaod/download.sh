huggingface-cli download shibing624/sharegpt_gpt4  --repo-type dataset  --local-dir /home/ubuntu/data/dataset/sharegpt4 > download_dataset.log 2>&1

huggingface-cli download  --repo-type model   --local-dir /home/ubuntu/data/model/Qwen-7B   Qwen/Qwen-7B



export RAY_DEDUP_LOGS=0 && export HEAD_NODE_IP=36.103.199.200

ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=36.103.199.235
ray start --address=36.103.199.235:6379 --node-ip-address=43.143.179.203

python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /home/ubuntu/data/model/Qwen-7B \
--worker-use-ray --max-model-len 4096 \
--dispatch-policy load --trust-remote-code \
--dispatch-load-metric remaining_steps \
--request-migration-policy SR --migration-backend gloo \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 3

python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /home/ubuntu/data/model/Qwen-7B \
--worker-use-ray --max-model-len 4096 \
--dispatch-policy rr --trust-remote-code \
--request-migration-policy SR --migration-backend gloo \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 3


export RAY_DEDUP_LOGS=0 && python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /home/ubuntu/data/model/Qwen-7B \
--worker-use-ray --max-model-len 4096 \
--dispatch-policy load --trust-remote-code \
--dispatch-load-metric virtual_usage \
--request-migration-policy SR --migration-backend gloo \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 3