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
### Data
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