export K=10 # LSH hyper-parameter for MagicPIG and Page Size for Quest
export CUDA_VISIBLE_DEVICES=2
export sink=4 # sink token
export local=64 # local token
export model=7 # 0: MagicPIG; 1: Quest; 2: TopK 3: Oracle Sampling
export expid=0
export L=256 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
export L=512 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
export L=1024 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid

export K=0
export L=256 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
export L=512 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid
export L=1024 # LSH hyper-parameter for MagicPIG and number of selected pages for Quest
bash run.sh llama3-8b-chat-128k synthetic $K $L $sink $local $model $expid