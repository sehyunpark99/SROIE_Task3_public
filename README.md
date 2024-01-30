

<h1 align="center">SROIE Task 3: Key Information Extraction</h1>

## Summary

The goal of this task is to extract entities from OCR images of the SROIE dataset. While the code for training, inference, evaluation, and preprocessed data has been given in advance, the priority of this task is primarily on analyze and tackle the limitation of the model. I first did some literature work on a key information extraction task of the SROIE dataset and how current SOTA achieved its performance via fine-tuning, hyper-parameter tuning, and architectural novelty. My final model is based on IIT-CDIP pre-trained LayoutLMV3-base, which is SOTA on most of the Document Image tasks. 

## Experimental Result

Here are the Experimental results. The experiments were conducted using two RTX3090 GPUs

### Model Selection: LayoutLMV1 vs LayoutLMV3
This is a process of selecting a model for SROIE key extraction. Based on the literature review, the current SOTA of this task is LayoutLMV2-Large, which is designed for the interaction among text, layout, and images in a multi-model framework [2]. However, I was not able to implement it due to lack of memory space on my server and incompatibility of [installation](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). Instead, I decided to evaluate LayoutLM and LauoutLMv3. LayoutLMv3 is a multi-modal foundation model for Document AI pre-trained by 11M Documents. Despite its simple architecture, it can well perform for both text-centric tasks and image-centric tasks [1]. 

As given in the result below, LayoutLMV3 outperforms LayoutLMV1. 

**Hyperparameters** 
> max_step = 1000, learning rate = 1e-5

**Model Specification**
> Both models were pre-trained on IIT-CDIP Test Collection 1.0 dataset with 11M documents. \
> LayoutLMV1: layoutlm-based-uncased (113M parameters) \
> LayoutLMV3: layoutlmv3-base (133M parameters) 

| LayoutLMV1 | LayoutLMV3 |
| ----------- | ----------- |
|![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/04c2d0db-02cb-4242-84fa-d2b11fd8a23b)| ![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/34336e9b-d265-4809-8bc4-d79b4eda7a89)|

### Hyperparameter Optimization
After selecting the appropriate model, the hyper-parameters were tuned for better performance. For evaluating optimal hyper-parameter, the code for evaluation of the test set was used. I have refered to these papers: [1](https://arxiv.org/pdf/1803.09820.pdf), [2](https://arxiv.org/pdf/2012.14740v4.pdf), [3](https://arxiv.org/pdf/1708.07120.pdf), [4](https://arxiv.org/pdf/2312.04528.pdf)

**Hyperparameters** 
> num_train_epochs = [10, 20, 50, 100, 200] \
> max_step = [1000, 1500] \
> learning_rate = [1e-5, 5e-5, 1e-4] \
> weight_decay = [0.0, 0.1] 

| Hyper-parameter | Table of Results |
| ----------- | ----------- |
|num_train_epochs|![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/2263448a-78ad-4178-91ef-806b676ceb9c)|
|max_step| ![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/6eb01f10-e54c-45a1-ac85-5628862e1a7b)|
|learning_rate| ![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/2829acba-1467-4207-a006-646a948b46c4)|
|weight_decay| ![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/d8ea1539-3d34-4180-a9a8-02988204b9fc)|

#### Comment
As a result of intensive experiments, the selected hyper-parameters are...\
**num_train_epochs = 100** \
**max_step = 1500** \
**learning_rate = 5e-5** \
**weight_decay = 0.0** 

## What I have tried
### Fine-tuning: Model Architecture
I tried to amend the last layer (classifier) of LayoutLMv3 to evaluate the effect of changes in the layer on the performance. In past [project](https://github.com/ChaiEunLee/apply-synthetic-medMNIST), I conducted experiments on how amendments in fc layers of ResNet-50 and DeiT affect the performance. However, while I was implementing the evaluation code, there was a mismatch in the number of gt_parse and pr_parse, which means it failed to accurately predict the token. For further studies, I want to succeed in changing the architecture of the model. 

| train.py | inference.py |
| ----------- | ----------- |
|![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/655f61ae-f476-452f-a2b4-ca8fef2b5cb1)|![image](https://github.com/sehyunpark99/SROIE_Task3/assets/37622900/689a39d0-c912-471b-ab58-6f5f9a874cb6)|

## Instructions
### Code Structure   
* ```requirement.text``` : To install required packages
* ```00_Tests.ipynb``` : Log of experiments
* ```01_Preprocessing_Tests.ipynb``` : Tests on preprocessing of SROIE dataset
* ```train.py``` edit version of given train.py code for logging, fine-tuning
* ```inference.py``` edit version of given inference.py code for logging, fine-tuning
* ```evaluation.py``` edit version of given evaluation.py code for logging, fine-tuning
* ```utils_1.py``` edit version of given utils.py code for logging, fine-tuning
* ```run.sh``` bash file for running multiple experiments

### Installation and Requirements
* Clone the repository
```
git clone https://github.com/sehyunpark99/SROIE_Task3.git
cd SROIE_Task3
```
* Install dependencies
```
conda env create -n sroie_task3 python=3.8.12
conda activate sroie_task3
pip install -r requirements.txt
```
* Prepare datasets: SROIE dataset & Noramlized via following links
[SROIE Dataset](https://paperswithcode.com/dataset/sroie),
[SROIE Paper](https://arxiv.org/pdf/2103.10213.pdf),
[LayoutLM](https://www.kaggle.com/code/urbikn/layoutlm-using-the-sroie-dataset/notebook)

### Run File
#### Training
**1) Using run.sh**   
- Fill in the experimental settings in run.sh
```
    bash run.sh
```

**2) Using train.py**
```
    python train.py --model_type layoutlmv3 --model_name_or_path 'microsoft/layoutlmv3-base' --overwrite_model_dir --do_train --num_train_epochs 10 --max_steps 1000 --warmup_steps 10000 --weight_decay 0.01 --learning_rate 0.001
```

#### Inference
**1) For test**
```
    python inference.py \
      --model_name_or_path 'microsoft/layoutlmv3-base' \
      --model_type layoutlmv3 \
      --do_predict \
      --do_lower_case \
      --max_seq_length 512 \
      --mode test \
      --model_dir './model/{PATH}' \
      --overwrite_output_dir \
      # --fine_tuning True
```
**2) For op_test**
```
    python inference.py \
      --model_name_or_path 'microsoft/layoutlmv3-base' \
      --model_type layoutlmv3 \
      --do_predict \
      --do_lower_case \
      --max_seq_length 512 \
      --mode op_test \
      --model_dir './model/{PATH}' \
      --overwrite_output_dir \
      # --fine_tuning True
```

## Approach

### Dataset
	Scanned Receipt OCR and Information Extraction (SROIE)
	- Task 1: Scanned Receipt Text Localization 
		○ Goal: To accurately localize texts with 4 vertices
		○ GT: at least at the level of words
		○ Evaluation
			§ Participants  apply different localization algorithms to locate text at different levels
			§ Evaluate based on DetVal + mAP + average recall --> F1 Score for ranking
	- Task 2: Scanned Receipt OCR
		○ Goal: To accurately recognize the text in a receipt image (no need for localization)
		○ Task: Need to offer a list of words recognized in the image (Latin & numbers)
		○ GT: list of words that appeared in the transcriptions --> need to tokenize all strings splitting on space
		○ Evaluation
			§ List of words compared with GT 
			§ Metrics: Precision (# correct/all detections), Recall (# correct/GT) --> F1 score for ranking
	- Task 3: Key Information Extraction from Scanned Receipts **[My Task]**
		○ Goal: To extract texts of several key fields from given receipts & save the texts for each receipt image in a JSON file
		○ GT: Information of content and category of the extracted text
		○ Evaluation
			§ Correct if the content & category of the extracted text match the GT
			§ Metric: mAP + recall --> F1 score for ranking

### Motivation
	- Currently: OCR becoming mature for many practical tasks 
	- Challenges: receipts OCR requires higher accuracy in performance than those of many commercial applications
		○ Especially when the receipt has low quality
		○ Document layout analysis entity recognition
	- What SROIE has
		○ A large-scale, well-annotated invoice datasets that are publicly available
		○ Special features: complex layouts & to address potential privacy issues
		○ Three tasks to be used in
		○ Comprehensive evaluation method 

### Possible Approaches
    - Data Processing
        ○ This experiment focuses more on the model side of engineering as normalized data has been already given, however, data augmentation, feature engineering, and data preprocessing are always good options to further improve the performance of the model. 
	- Model architecture engineering
        ○ I have tried to amend the model architecture of LayoutLMv3, however, there were some incompatibility in evaluation and inference code. For further studies, I would like to change the code and customize the architecture so that I can further improve the performance of the model. 



## Citation
[1] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In Proceedings of the 30th ACM International Conference on Multimedia
(MM ’22), October 10–14, 2022, Lisboa, Portugal. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3503161.3548112 \
[2] Xu, Yang, et al. "Layoutlmv2: Multi-modal pre-training for visually-rich document understanding." arXiv preprint arXiv:2012.14740 (2020).
