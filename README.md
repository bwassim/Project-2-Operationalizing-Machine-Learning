# Operationalizing Machine Learning
## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we will continue to work with the [`Bank Marketing Dataset`](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). We will use Azure to configure a cloud based machine learning production model, deploy it, and consume it. We will also create, publish and consume a pipeline. 
The project main steps are depicted in the following diagram

<img src="./images/diagram-project-2.png">


## Authentication
This step consists in the creation of a Service Principal (SP) for accessing Azure workspace. Since the provided lab holds insufficient privileges for this step, it was not executed. However in general use case, we will follow these steps: 

* Ensure the az command-line tool is installed along with the ml extension

The Azure Machine Learning extension allows us to interact with Azure Machine Learning Studio, part of the az command.
* Ensure it is installed with the following command:
```
   az extension add -n azure-cli-ml
```
Create the Service Principal with az after login in
```
az ad sp create-for-rbac --sdk-auth --name ml-auth
```
Capture the "objectId" using the clientID:
```
az ad sp show --id xxxxxxxx-3af0-4065-8e14-xxxxxxxxxxxx
```
Assign the role to the new Service Principal for the given Workspace, Resource Group and User objectId
```
$ az ml workspace share -w Demo -g demo --user xxxxxxxx-cbdb-4cfd-089f-xxxxxxxxxxxx --role owner
```
## Automated ML Experiment
This part can be organised in the following sections 
### 1 - Data
We try to load the training dataset `bankmarketing_train.csv` from the workspace. Otherwise we create it from the file.
```python
found = False
key = "BankMarketing Dataset"
description_text = "Bank Marketing DataSet for Udacity Course 2"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)
```

<img src="./images/data-creation.png">

### 2 - AutoML config 
Create a general AutoML settings object

```python 
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
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
```
### 3 - Create a Pipeline and AutoML step
```python
from azureml.pipeline.core import PipelineData, TrainingOutput

ds = ws.get_default_datastore()
metrics_output_name = 'metrics_output'
best_model_output_name = 'best_model_output'

metrics_data = PipelineData(name='metrics_data',
                           datastore=ds,
                           pipeline_output_name=metrics_output_name,
                           training_output=TrainingOutput(type='Metrics'))
model_data = PipelineData(name='model_data',
                           datastore=ds,
                           pipeline_output_name=best_model_output_name,
                           training_output=TrainingOutput(type='Model'))
```
Create AutoML step 

```python
automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)
```
Create the Pipeline and Run it
```python
from azureml.pipeline.core import Pipeline
pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])

pipeline_run = experiment.submit(pipeline)
```
In this experiment, our goal is to solve a classification problem. As mentioned in the first project, we seek to classify whether a certain client will subcsribe to a term deposit. The output label given in the last column `y` holds all the decisions made by different clients.

The figure below shows the results of our Run

<img src="./images/run-autoML.png">

One can notice that that VotingEnsemble is the algorithm that provides the best metric results. When clicking on the Votingensemble we observe the details given in the figure below

<img src="./images/run-best-voting.png">

### 4 - Pipeline Run Overview 
Moving to the experiment section on Azure studio, one can find the detail of the pipeline we've just run. 
<img src="./images/pipeline-run-overview.png">

### 5 - Best model saved and tested
We see that we have trained several model through AutoML and the best model is saved and is ready to be deployed. 

## Model Deployment 
Deploying the Model will allow to interact with HTTP API service and the model by sending data over POST requests.
Notice in the figure that clicking on the deploy button will open up the window on the right handside. We enable authentication and make sure to deploy the model using Azure COntained Instance (ACI)

<img src="./images/model-deployment.png">

## Enabling Application Insights
There are two ways of enabling Application Insights. On can click on `Advanced` at the botttom of the right handside of the above figure and enable Application Insights before deploying the model. It can also be done using our log.py file with the following command line 
```python 
service.update(enable_app_insights=True)
```

<img src="./images/rest-endpoint-model-app-insight.png">

Running the `log.py` on the terminal will enable Application Insights together with logging information. One can observe the application insights url on the left handside of the above figure. 
> It's important to note that in order for the log.py script to run correctly, it is important to download the `config.json` file from Azure studio and put it in the same folder. 

## Swagger Documentation
Swagger is a tool that helps build, document, and consume RESTful web services like the ones we are deploying in Azure ML Studio. It further explains what types of `HTTP requests` that an API can consume, like `POST` and `GET`. 

* Ensure Docker is installed on our computer.

Azure provides a `Swagger JSON file` for deployed models. The swagger URI is used to download the JSON file. 

1. `Swagger.sh` will download the latest Swagger container, and it will run it on port 80. In our case since no premisson was granted for port 80, we updated the script to a port with higher value : 9000.
2. `serve.py` will start a Python server on port 9000. The file needs to be in the same directory as our `swagger.json`

<img src="./images/best-model-swagger.png">
<img src="./images/swagger-1.png">
<img src="./images/swagger-2.png">

## Consume Model Endpoint
* Once the model is deployed, we need to use the `endpoint.py` script to interact with the trained model. 
* We also need to modify the `scoring_uri` and `key` to match the key and the REST endpoint URI that was generated after deployment.


<img src="./images/endpointpy.png">
<img src="./images/testendpointpy.png">

## Benchmark
It's interesting to create a baseline for to have an idea on the request's response time. Basically we are interested to know how many requests per second the endpoint is capable of serving or similarly the average response time.
The `benchmark.sh` script inlcudes mainly the apache benchmark command `ab` that runs against the selected endpoint using `data.json` file created by the `endpoint.py` we used previously. 

```
ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer 9h0CCgKtYsVlBOQ3USjvkADAgxD8qLsV' http://bbb33f05-0a41-4661-80a6-9154d7249855.southcentralus.azurecontainer.io/score
```
<img src="./images/benchmark.png" >
<img src="./images/benchmark-2.png" >

## Create, Publish and Consume a Pipeline