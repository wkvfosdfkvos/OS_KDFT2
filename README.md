
<h4 align="right">
    <p>
        | <b>English</b> |
        <a href="https://github.com/Jungwoo4021/OS-KDFT/blob/main/readme/README_ko.md">한국어</a> |
    </p>
</h4>

<h1 align="center">
    <b>OS-KDFT2</b>
</h1>

<h2 align="center">
    <b>One-step joint training strategy for a large pre-trained model compression and target task fine-tuning to get a compressed target model
</h2>

<h3 align="left">
	<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=Python&logoColor=white"></a>
	<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html#rel-23-08"><img src="https://img.shields.io/badge/23.08-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
	<a href="https://huggingface.co/"><img src="https://github.com/Jungwoo4021/OS-KDFT/blob/main/readme/icon_hugging_face.png?raw=true"></a>
	</p>
</h3>

# Introduction
Pytorch code for following paper:
* **Title** : FURTHER IMPROVEMENT TO ONE-STEP KNOWLEDGE DISTILLATION AND TARGET TASK FINE-TUNING
* **Autor** :  Jungwoo Heo, Chan-yeong Lim, Ju-ho Kim, Hyun-seo Shin, Jae-han Park, and Ha-Jin Yu

## About OS-KDFT2
One-step knowledge distillation and fine-tuning (OS-KDFT2) is a training strategy that performs an audio pre-trained large model (such as wav2vec2.0, HuBERT) compression and fine-tuning to get a compressed task-specialized model. 

## Paper abstract
Self-supervised learning (SSL) models have become key factors that derive state-of-the-art performance in speech signal processing. However, their vast amount of parameters makes development and deployment challenging. To tackle these problems, several studies leverage knowledge distillation, which utilizes knowledge from a pre-trained large model to build a lightweight model that produces similar results. However, traditional light-weighting approaches for SSL model performed model compression and target task tuning independently, resulting in suboptimal performance or substantial computational costs. The author's previous research suggested one-step knowledge distillation and fine-tuning (OS-KDFT), which simultaneously conducts knowledge distillation and fine-tuning to transfer task-tailored features. Despite the success of OS-KDFT, there are two areas for improvement: domain mismatch problems in multi-task learning with different datasets and emphasizing features for target tasks. In this paper, we developed OS-KDFT2, which modifies the data sampling strategy to gradually increase the proportion of the target dataset to optimize the model for the target dataset domain. Furthermore, we improve the structure of the student model by employing attention mechanisms. OS-KDFT2 achieved better performance compared to previous methods while maintaining comparable computational costs in Speech processing Universal PERformance Benchmark and VoxCeleb2. 
Our experimental sources are available on GitHub 

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:23.08-py3' image of Nvidia GPU Cloud for conducting our experiments. 

* Python 3.10

* Pytorch 2.1.0a0+29c30b1

* Torchaudio 2.0.1

# Datasets
While the experiments in the paper were performed with reference to the SUPERB benchmark, the experiments in this repository use the 'VoxCeleb1' and 'LibriSpeech' dataset, as it is code that performs speaker verification experiments. 

# Run experiment

### STEP1. Set system arguments
First, you need to set system arguments. You can set arguments in `arguments.py`. Here is list of system arguments to set.

```python
1. 'path_log': path of saving experiment logs.
    CAUTION!! 
        If a directory already exists in the path, it remove the existing directory.

2. 'path_libri': path where LibriSpeech dataset is stored.

3. 'path_train': path where VoxCeleb1 train partition is stored.

4. 'path_test': path where VoxCeleb1 test partition is stored.

5. 'path_trials': path where Vox1-O, Vox1-E, Vox1-H trials is stored.

5. 'usable_gpu': number of GPUs to use in the experiment. 

```

### STEP2. Set system arguments
### Additional logger
We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`
```python
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_user'   : 'user-name',
'wandb_token'  : 'WANDB_TOKEN',

'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```

2. In `main.py`

```python
# Just remove "#" in logger

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        #).use_wandb(args['wandb_user'], args['wandb_token'] <- here
        #).use_neptune(args['neptune_user'], args['neptune_token'] <- here
        ).build()
```

### STEP3. RUN
Run main.py in scripts.

```python
>>> python main.py
```
