# Readme for black-box optimization development version

More details are provided in our ICML paper [Black-Box Tuning for Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514) and our arxiv paper [BBTv2: Pure Black-Box Optimization Can Be Comparable to Gradient Descent for Few-Shot Learning](https://arxiv.org/abs/2205.11200).

## Prepare your environment

The implementation of Black-Box Tuning is quite simple, you can check our code and easily implement it in your own environment. Or you can create a new environment to run our implementation, which is based on `pycma`, `Transformers` and `FastNLP`. Optionally, you can use `fitlog` to monitor experimental results. You can uncomment the fitlog-related lines in our code to use it.

```bash
conda create --name bbt python=3.8
conda activate bbt
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
git clone https://github.com/txsun1997/Black-Box-Tuning
cd Black-Box-Tuning
```

## Running baseline

Now you can run Black-Box Tuning with `run.sh`:

```bash
bash run.sh
```


```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100
```


## Inference Optimization

In contrast to training with gradient descent, BBT (and BBTv2) only requires model forward computation, and therefore can be significantly accelerated using [ONNX Runtime](https://onnxruntime.ai/) or NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt). 

Here we provide an implementation of inference optimization using ONNX Runtime. You can obtain ~2x speedup using only one line of code.

SDK `onnxruntime-gpu` is required for optimization. Installation of this package can be troublesome. And there may be some environment-specific errors or unexpected performance. But in real-world scenarios, this is a part of the black box on the server side.

On an NVIDIA GeForce RTX 3090 GPU with Driver Version: 470.82.00, CUDA Version: 11.4 and Cudnn Version:8.2.4, the following code works well to configure the environment.

See [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for detailed hardware requirements.
```bash
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install onnx
pip install onnxruntime-gpu==1.10.0
pip install coloredlogs
pip install sympy
```

To export a BBT model based on `PyTorch` to an `ONNX` model, 
you can run `export_and_optimize.py` with all arguments set to default to get a demo onnx model.

```bash
python export_and_optimize.py
```
Two models will be saved to `./onnx_models/`, namely exported (not accelerated) and optimized model.
Then you can modify `run.sh`. 
By setting parameter `inference_framework` to `'ort'` and `onnx_model_path` to `<Your model path>`,
a faster version of BBT is ready. Here is an example.
```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 \
  --inference_framework 'ort' \
  --onnx_model_path './onnx_models/optimized_model.onnx'
```

To add some flexibility to model optimization, we provided some options in `export_and_optimize.py`.
You can adjust these arguments in `export_and_optimize.sh`. Here is an example.
```bash
python export_and_optimize.py \
  --batch_size 32 \
  --max_seq_len 128 \
  --n_prompt_tokens 50 \
  --prompt_embed_dim 1024 \
  --cat_or_add "add" \
  --exported_model_name 'model' \
  --optimized_model_name 'optimized_model'
```

## Try your own optimization algorithm

In this branch, we wrapped BBT in a 3-part logic: arguments-LMForwardAPI-optimization. They are implemented in arguments.py, bbt.py and shallow_bbt_optim.py, respectively.
We can completely neglect the middle part and just treat it like a black box.
Since the meaning of arguments is straightforward to understand, we can focus on the optimization part.

There is a note in line 62-69 in shallow_bbt_optim.py, before which all we need to modify(or add if we need) are hyperparameters.
And we can implement our algorithm from then on. We provide a baseline example.

We offer 3 datasets for evaluation, namely sst2(sentiment analysis, 2-label), agnews(topic cls, 4-label) and snli(sentence-pair cls, 3-label).
Difficulty of the 3 datasets increases in order. So their alias in the code are `easy`, `medium` and `hard`.

We give a baseline result reported in the BBT paper.

## Baseline performance

sst2(easy): 89.56(0.25)

agnews(medium): 81.51(0.79)

snli(hard): 46.58(1.33)
