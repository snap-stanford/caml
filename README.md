# Zero-shot causal learning


******NOTE: THIS REPOSITORY IS EXTREMELY MESSY AND WILL BE FULLY CLEANED PRIOR TO PUBLICATION*****. W

While it is functional, it is not pretty at this point, we will clean it up and encapsulate after the submission. Also, the branch for LINCS setting in the LINCS_BRANCH folder. 


## Experiments 
All our experiment configurations can be found under `experiments/`

For instance, to run the hyperparameter search for the proposed model, CaML, we create a Weights & Biases sweep using the yaml file: 
`experiments/full_model.yaml` and then start the sweep agent.  
Alternatively, a single model fit can be started by executing the file `experiments/full_model.sh`, however the `.sh` files are just for illustrative purposes and the exact configuration may not be up-to-date, as the .yaml files were used for all experiments. 

Our experiment folder contains numeric results under `experiments/results`.  
Under `experiments/results/claims` all the evaluation metrics for all methods in the Claims dataset are logged.  
Under `experiments/results/lincs` all the evaluation metrics for all methods in the LINCS dataset are logged.  


## Training a model:  

To train a model on the Claims Dataset, we use the train_caml.py script, example below. Note, however that the Claims dataset is private and can not be released at this point. 

``$ python metalearn/train_caml.py --batch_norm=true --batch_size=8192 --dropout=0 --l1_reg=0.000001 --learning_rate=0.001 --loss=mse --meta_batch_size=4 --meta_learning_rate=2 --meta_learning_rate_final_ratio=0.1 --model_dim=256 --n_iterations=1000 --n_layers=6 --caml_k=50 --step=tau --task_embedding_config=early_concat --use_lr_scheduler=false --use_task_embeddings=true --val_interval=5 --weight_decay=0.005 ``  

Key parts of this are outlined below: 
- Lines (approx 69-76) starting with ``exclusionLoader = QueryPatientsByDrug()``: here we load the validation and testing drugs, and get the patient IDs of individuals who took these drugs, so we can hide them from the training (and hide testing drug patients from the validation) 
- Lines 98-104 starting with ``for val_task in val_tasks`` Here we load the validation tasks (drugs) and store them for zero-shot validation during training 
- Lines 132-135 starting with ``te_mapper = TaskEmbeddingMapper`` This class takes as input the task ID (DrugBank ID) and returns the task (drug) embedding. This is later fed into the model
- Lines 149-194 starting with ``if use_task_embeddings:``. There are currently three options 1) Don't use task embeddings 2) Concatenate the task embeddings at the very beginning of the neural network as ordinary features (models.MLPModel) 3) Concatenate the task embeddings later in the neural network after we have applied some encoding to the patient-level information (models.MLPLateConcatModel). Here depending on the hyperparameters we initialize one of these 3 models.
- Lines 280-323 starting with ``sampled = df_tasks.sample(n=1,weights=sample_weights).squeeze()``. Here we 1) sample a task from the list of all tasks (tasks are a sinle or combination of drugs 2) Process the task to calculate the \tau label (the estimate of the causal effect) and filter for only patients who received the drug (W=1). Here note that X is a sparse matrix
- Lines 325-342 starting with ``if use_task_embeddings:`` Here we create the associated dataset and dataloader for the task. Note that if we are using task embeddings the ClaimsTaskDataset class separates the sparse patient-level features (X_train_1) from the dense drug-level features (task_embedding) so that we have the option to concatenate them later on
- Lines 391-460 starting with ``if iteration%args.val_interval==0:`` Here we validate the zero-shot performance of our model on a set of unseen validation drugs. We loop through the drugs, generate predictions, and then measure performance using 2 main metrics (RATE, Recall)

``$ (NOTE: the exact line numbers may be out of sync with latest code stage, but the explanations may still be useful) 

While our models are currently initialized inside the train_caml.py script depending on experimental configurations, `src/models/caml.py` contains a model class pointing to the building blocks of our method. (We intend to clean all methods to have separate model classes with model-specific parsers). 

## Environment

### Claims setting
`virtualenv caml_env` to start virtual_env named caml_env  

`source caml_env/bin/activate` to activate the env  

`source caml_env/install.sh` to install all the libraries (also cuda-version torch etc)

### LINCS setting
WIP (#TODO: add env setup)




