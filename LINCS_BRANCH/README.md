# ehr-drug-sides




## Preprocessing:  

To load and reorder the full dataset (to account for an artefact from BigQuery), cd to `preprocessing` and run:   

``$ python load_reorder.py --n_jobs=50 --n_chunks=500 &> reorder_log.txt``  

## Training zero shot model:  

To train a model, we use the train_caml.py script, example below:   

``$ python metalearn/train_caml.py --batch_norm=true --batch_size=8192 --dropout=0 --l1_reg=0.000001 --learning_rate=0.001 --loss=mse --meta_batch_size=4 --meta_learning_rate=2 --meta_learning_rate_final_ratio=0.1 --model_dim=256 --n_iterations=1000 --n_layers=6 --caml_k=50 --step=tau --task_embedding_config=early_concat --use_lr_scheduler=false --use_task_embeddings=true --val_interval=5 --weight_decay=0.005 ``  

Key parts of this are outlined below: 
- Lines (approx 69-76) starting with ``exclusionLoader = QueryPatientsByDrug()``: here we load the validation and testing drugs, and get the patient IDs of individuals who took these drugs, so we can hide them from the training (and hide testing drug patients from the validation) 
- Lines 98-104 starting with ``for val_task in val_tasks`` Here we load the validation tasks (drugs) and store them for zero-shot validation during training 
- Lines 132-135 starting with ``te_mapper = TaskEmbeddingMapper`` This class takes as input the task ID (DrugBank ID) and returns the task (drug) embedding. This is later fed into the model
- Lines 149-194 starting with ``if use_task_embeddings:``. There are currently three options 1) Don't use task embeddings 2) Concatenate the task embeddings at the very beginning of the neural network as ordinary features (models.MLPModel) 3) Concatenate the task embeddings later in the neural network after we have applied some encoding to the patient-level information (models.MLPLateConcatModel). Here depending on the hyperparameters we initialize one of these 3 models.
- Lines 280-323 starting with ``sampled = df_tasks.sample(n=1,weights=sample_weights).squeeze()``. Here we 1) sample a task from the list of all tasks (tasks are a sinle or combination of drugs 2) Process the task to calculate the \tau label (the estimate of the causal effect) and filter for only patients who received the drug (W=1). Here note that X is a sparse matrix
- Lines 325-342 starting with ``if use_task_embeddings:`` Here we create the associated dataset and dataloader for the task. Note that if we are using task embeddings the ClaimsTaskDataset class separates the sparse patient-level features (X_train_1) from the dense drug-level features (task_embedding) so that we have the option to concatenate them later on
- Lines 391-460 starting with ``if iteration%args.val_interval==0:`` Here we validate the zero-shot performance of our model on a set of unseen validation drugs. We loop through the drugs, generate predictions, and then measure performance using 2 main metrics (RATE, Recall)

``$ python load_reorder.py --n_jobs=50 --n_chunks=500 &> reorder_log.txt``  


## Environment

### older version: 
`poetry install`  

For cuda torch, I then used:  

`pip install torch==1.7.1 --extra-index-url https://download.pytorch.org/whl/cu113` 

### env for hypernetwork and EHR stuff ():  

`virtualenv caml_env` I started a virtual_env named caml_env  

`source caml_env/bin/activate` to activate the env  

`source caml_env/install.sh` to install all the libraries as I did

To run a (residual) hypernetwork, run:  
`python metalearn/train_hypernet_caml.py`  
or create a wandb sweep using this script as the program 

wandb agent dsp/dl-zero-shot-cate/


parallel --jobs 4 "CUDA_VISIBLE_DEVICES={} wandb agent dsp/lincs/hzdhkgaq > out{}1.txt" ::: {0..3}
parallel --jobs 4 "CUDA_VISIBLE_DEVICES={} wandb agent dsp/lincs/hzdhkgaq > out{}2.txt" ::: {0..3}





nohup wandb agent dsp/lincs/h7njmo3a &
nohup wandb agent dsp/lincs/wgffztqv &
nohup wandb agent dsp/lincs/wgffztqv &
nohup wandb agent dsp/lincs/wgffztqv &
nohup wandb agent dsp/lincs/wgffztqv &


nohup wandb agent dsp/lincs/wgffztqv > /dev/null 2>&1& &
