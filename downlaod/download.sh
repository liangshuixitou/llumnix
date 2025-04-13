huggingface-cli download shibing624/sharegpt_gpt4  --repo-type dataset  --local-dir /data/dataset/sharegpt4 
huggingface-cli download  --repo-type model   --local-dir /data/model/Qwen2.5-3B   Qwen/Qwen2.5-3B



export RAY_DEDUP_LOGS=0 && export HEAD_NODE_IP=36.103.199.200

ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=36.103.199.235
ray start --address=36.103.199.235:6379 --node-ip-address=43.143.179.203

ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=10.212.70.38
ray start --address=10.212.70.38:6379

export RAY_DEDUP_LOGS=0 && rm -f vllm.serve.load.log && python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /data/model/Qwen2.5-3B \
--worker-use-ray --max-model-len 4096 \
--dispatch-policy load --trust-remote-code \
--dispatch-load-metric remaining_steps \
--request-migration-policy SR --migration-backend gloo \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 3 \
2>&1 | tee vllm.serve.load.log


export RAY_DEDUP_LOGS=0 && rm -f vllm.serve.rr.log&& python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /data/model/Qwen2.5-3B \
--worker-use-ray --max-model-len 4096 \
--dispatch-policy rr --trust-remote-code \
--request-migration-policy SR --migration-backend gloo \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 3 \
2>&1 | tee vllm.serve.rr.log



export RAY_DEDUP_LOGS=0 && rm -f vllm.serve.vu.log && python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /data/model/Qwen2.5-3B \
--worker-use-ray --max-model-len 4096 \
--dispatch-policy load --trust-remote-code \
--dispatch-load-metric virtual_usage \
--request-migration-policy SR --migration-backend gloo \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 3 \
2>&1 | tee vllm.serve.vu.log


export RAY_DEDUP_LOGS=0 && rm -f vllm.serve.mig.log && python -m llumnix.entrypoints.vllm.serve --host 0.0.0.0 \
--port 37000  --model /data/model/Qwen2.5-3B \
--enable-migration \
--worker-use-ray \
--max-model-len 4096 \
--dispatch-policy load --trust-remote-code \
--request-migration-policy SR --migration-backend rayrpc \
--migration-buffer-blocks 32 --tensor-parallel-size 1 \
--request-output-queue-port 38234 \
--max-num-batched-tokens 16000 \
--enable-port-increment --max-instances 2 \
2>&1 | tee vllm.serve.mig.log


export CUDA_VISIBLE_DEVICES=1
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=10.212.70.38 --num-gpus=1
ray start --address=10.212.70.38:6379