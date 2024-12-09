export K=10 # LSH hyper-parameter for MagicPIG and Page Size for Quest
export CUDA_VISIBLE_DEVICES=5
export sink=4 # sink token
export local=16 # local token
export model=7 # 0: MagicPIG; 1: Quest; 2: TopK 3: Oracle Sampling
export expid=0
export K=0

export L=32 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
export L=64 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
export L=128 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
