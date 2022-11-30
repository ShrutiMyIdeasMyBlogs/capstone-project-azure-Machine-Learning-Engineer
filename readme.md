# Project Title
In this project I have created two classification models on bankmarketing one using automl in python SDK and another one using Hyperdrive(Hyperparameter Tunning)

## Dataset
I am using the bankmarketing data of direct marketing campaigns of a banking institution. This dataset identify whether the product was purchased or not based on various attributes.
### Overview
The data set of bank marketing has been taken from UCI https://archive.ics.uci.edu/ml/datasets/Bank+Marketing. Two classification models have been created,one using automl in python SDK and another one using Hyperdrive(Hyperparameter Tunning)
The model is created based on following attributes:
1. age
2. job
3. marital
4. education
5. default
6. housing
7. loan
8. contact
9. month
10. day_of_week
11. duration
12. campaign
13. pdays
14. previous
15. poutcome
16. emp.var.rate
17. cons.price.idx
18. cons.conf.idx
19. euribor3m
20. nr.employed

### Task
Two classification models have been created,one using automl in python SDK and another one using Hyperdrive(Hyperparameter Tunning)

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
The following configuration is used for AutoML

automl_settings = {
    "experiment_timeout_minutes": 60,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )    

### Results
VotingEnsemble model has outperformed the other models.Following is the metrics details:
{'weighted_accuracy': 0.9653231101734875,
 'average_precision_score_weighted': 0.9539762342704304,
 'precision_score_micro': 0.9201820940819423,
 'AUC_macro': 0.9424651799491338,
 'f1_score_weighted': 0.9143989541642051,
 'average_precision_score_micro': 0.9802616591730408,
 'recall_score_micro': 0.9201820940819423,
 'norm_macro_recall': 0.4767239606777476,
 'recall_score_macro': 0.7383619803388738,
 'precision_score_macro': 0.8194160707149187,
 'balanced_accuracy': 0.7383619803388738,
 'f1_score_micro': 0.9201820940819423,
 'matthews_correlation': 0.5518573988547033,
 'log_loss': 0.27285025566709314,
 'accuracy': 0.9201820940819423,
 'recall_score_weighted': 0.9201820940819423,
 'AUC_micro': 0.9793832104098499,
 'AUC_weighted': 0.9424651799491338,
 'f1_score_macro': 0.7708311020316416,
 'average_precision_score_macro': 0.8205297606922544,
 'precision_score_weighted': 0.9126693178193268,
 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_f7741924-507a-4922-89f4-f28de06f64ee_51/confusion_matrix',
 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_f7741924-507a-4922-89f4-f28de06f64ee_51/accuracy_table'}

## Hyperparameter Tuning
The scikit-learn pipe;ine has been developed with the following details:
1. The parameter space is defined and Random Parameter Sampling is used, as follows
param_space={
    '--C':choice(0.001,0.01,0.1,1,10),
    '--max_iter':choice(range(1,100,10))
}
ps = RandomParameterSampling(param_space)

2 Specify the early termination Policy
policy = BanditPolicy(slack_amount=0.2,
                     evaluation_interval=1,
                     delay_evaluation=5)
					 
3. Defined the training script and default hyperparameter values in ScriptRunConfig  
src = ScriptRunConfig(source_directory="./",
                     script="train.py",
                     environment=sklearn_env,
                      compute_target=aml_cluster,
                     arguments=['--C',1.0,'--max_iter',100])

3. Create a HyperDriveConfig using the src object, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(run_config=src,
                                    hyperparameter_sampling =ps,
                                    primary_metric_name="Accuracy",
                                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                    max_total_runs=6,
                                    max_concurrent_runs=4)

### Results
Logisitic Regression was run with various values of hyperparemeter and the best accuracy score of 0.91335357 was achieved with value of C as 0.01 and max_iteration values 81

## Model Deployment
The Automl model has been deployed succesfully and the screenshot has been attached in the images folder.

## Screen Recording
Following is the screen recording.The voice in the recording is not available.The Video first shows the deployment of the model as Restend point,after that it has been consumed succesfully on the data.In the end service and compute have been deleted
https://youtu.be/vwNl55CvvxE
