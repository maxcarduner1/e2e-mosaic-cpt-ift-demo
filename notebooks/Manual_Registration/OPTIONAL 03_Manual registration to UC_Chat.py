# Databricks notebook source
# MAGIC %md # Manual Registration to UC
# MAGIC
# MAGIC We had a small incident where there was a bad MLflow version that prevents runs registered in UC from being deployed with Provisioned Throughput Model Serving. Instead of re-training your run, you can use the following code to manually re-register your model in UC correctly for deployment.
# MAGIC
# MAGIC Note: you'll need to use a compute instance that has enough memory to fit your model in RAM so it can download the model locally and then re-register the model to MLflow.
# MAGIC
# MAGIC If you don't have access to a compute instance of the above size, you'll have to re-run your training job. Instead of re-running the full experiment, with the help of `custom_weights_path`, you could load the previously trained checkpoint and then continue training on a small dataset for a short duration. The resulting new model will then automatically upload the final checkpoint and register your model as usual.

# COMMAND ----------

# MAGIC %sh pip install mosaicml-cli mlflow einops transformers pandas torch torchvision databricks-genai --upgrade

# COMMAND ----------

# MAGIC %pip install mlflow-skinny[databricks]

# COMMAND ----------

dbutils.library().restartPython()

# COMMAND ----------

# MAGIC %md ## Inputs
# MAGIC You only need to change the following constants. Then, run the rest of the notebook!

# COMMAND ----------

RUN_NAME='contd-pretrain-meta-llama-3-1-8b-tg1i9b'
EXPERIMENT_PATH = '/Users/max.carduner@databricks.com/contd-pretrain-meta-llama-3-1-8b-tg1i9b'
REGISTERED_MODEL_NAME = 'users.max_carduner.contd-pretrain-meta-llama-3-1-8b-tg1i9b'
MLFLOW_TRACKING_URI = 'databricks'
MLFLOW_REGISTRY_URI = 'databricks-uc'

# COMMAND ----------

from databricks.model_training import foundation_model as fm
run = fm.get(training_run=RUN_NAME)
MODEL_NAME = run.model

# COMMAND ----------

# MAGIC %md ## Function

# COMMAND ----------

# DBTITLE 1,MLflow Artifact Downloader
import mlflow
from mlflow import MlflowClient
from pathlib import Path

def download_artifacts(experiment_path: str, run_name: str) -> str:
  # Returns the parent path to the downloaded artifacts
  # Get the MLflow experiment by the given experiment path
  experiment = mlflow.get_experiment_by_name(name=experiment_path)
  # Get me the experiment run id given the experiment run name
  client = MlflowClient()
  runs = client.search_runs(experiment.experiment_id)
  for run in runs:
      if run.info.run_name == run_name:
          run_id = run.info.run_id
          break
  print(f"Found MLflow run with run_id: {run_id}")

  print("Start downloading files temporarily...")
  # Only download the config file and HF related artifacts
  all_artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
  subdir = mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=f'{run_name}/checkpoints/huggingface')
  local_path = ''
  for artifact in subdir:
    print(f"Downloading artifact at path: {artifact.path}")
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact.path)
  return Path(local_path)

# COMMAND ----------

import json, os

# For TPT compatibility
_MODEL_CONFIG_PATH = os.path.join('model', 'config.json')
_GENERATION_CONFIG_PATH = os.path.join('model', 'generation_config.json')
_CONFIG_KEYS_TO_DELETE = {
    '_name_or_path',
    'initializer_range',
    'output_router_logits',
    'router_aux_loss_coef',
    'router_jitter_noise',
    'transformers_version',
    'use_cache',
}
_GENERATION_CONFIG_KEYS_TO_KEEP = {
    'eos_token_id',
}
_CONFIG_KEYS_WITH_DEFAULTS = {
    'attention_dropout': 0.0,
    'emb_pdrop': 0.0,
    'embedding_dropout': 0.0,
    'resid_pdrop': 0.0,
    'residual_dropout': 0.0,
}

def pre_register_edit(local_save_path: str):
    model_config_path = os.path.join(local_save_path, _MODEL_CONFIG_PATH)
    generation_config_path = os.path.join(local_save_path,
                                          _GENERATION_CONFIG_PATH)

    if os.path.exists(model_config_path):
        print('Start parsing model config')
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)

        original_model_config_keys = list(model_config.keys())
        for key in original_model_config_keys:
            if key in _CONFIG_KEYS_TO_DELETE:
                del model_config[key]
            elif key in _CONFIG_KEYS_WITH_DEFAULTS:
                model_config[key] = _CONFIG_KEYS_WITH_DEFAULTS[key]

        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print('Finished parsing model config')

    if os.path.exists(generation_config_path):
        print('Start parsing generation config')
        with open(generation_config_path, 'r') as f:
            generation_config = json.load(f)

        original_generation_config_keys = list(generation_config.keys())
        for key in original_generation_config_keys:
            if key not in _GENERATION_CONFIG_KEYS_TO_KEEP:
                del generation_config[key]

        with open(generation_config_path, 'w') as f:
            json.dump(generation_config, f, indent=4)
        print('Finished parsing generation config')

# COMMAND ----------

# DBTITLE 1,Unified Catalog Model Logger
import mlflow, os, einops
from tempfile import TemporaryDirectory
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import mcli
from tempfile import TemporaryDirectory

# To register this model on Unity Catalog


def load_model_from_mlflow_and_log_to_uc(
    experiment_path: str, run_name: str, model_name: str
):
    with TemporaryDirectory() as tempdir:
        # Given the experiment path, download all the artifacts associated with the experiment
        parent_dir = download_artifacts(experiment_path, run_name)
        print("Finished downloading files temporarily. Now loading model...")

        # Parse the config files for TPT
        pre_register_edit(parent_dir)

        # Now that all files are downloaded, you can load the model and tokenizer from the tempdir
        tokenizer = AutoTokenizer.from_pretrained(parent_dir, trust_remote_code=True)
        print("Tokenizer loaded...")
        model = AutoModelForCausalLM.from_pretrained(parent_dir, trust_remote_code=True)
        print("Model loaded...")

        # Load model, we will then log the model in MLflow ensuring to add the metadata to the mlflow.transformers.log_model.
        # We may need to set our experiment. Comment this out if you know you won't need to.
        print("Creating Experiment")
        mlflow.set_experiment(experiment_path)

        # Log model to MLflow - This will take approx. 5mins to complete.
        # Define input and output schema

        input_schema = Schema(
            [
                ColSpec(DataType.string, "messages"),
                ColSpec(DataType.double, "temperature", optional=True),
                ColSpec(DataType.long, "max_tokens", optional=True),
            ]
        )

        output_schema = Schema([ColSpec(DataType.string)])

        print("Logging model with MLflow.")
        with mlflow.start_run() as mlflow_run:
            components = {
                "model": model,
                "tokenizer": tokenizer,
            }

            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path="model",
                input_example = {
                    "messages": [
                        {"role": "user", "content": "What is Apache Spark"},
                        ], # This input example is just an example
                        "temperature": [0.1],
                         "max_tokens": [5000],
                     },
                # input_example=
                
                # {
                #         "messages": [
                #             "what is mlflow?"
                #         ],  # This input example is just an example
                #         "temperature": [0.1],
                #         "max_tokens": [256],
                #     },
                task="llm/v1/chat",
                databricks_model_source="genai-fine-tuning",
                pretrained_model_name=MODEL_NAME,
                metadata={
                    "task": "llm/v1/chat"
                },  # This metadata is currently needed for optimized serving
                registered_model_name=model_name,
            )

            return mlflow_run


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
mlflow_run = load_model_from_mlflow_and_log_to_uc(
    experiment_path=EXPERIMENT_PATH, run_name=RUN_NAME, model_name=REGISTERED_MODEL_NAME
)
mlflow_run

# COMMAND ----------


