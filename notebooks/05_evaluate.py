# Databricks notebook source
# single node MLR 14.3, 16 cores
%pip install mlflow[genai,databricks]
dbutils.library.restartPython()

# COMMAND ----------

import json
# set params below
catalog = 'users'
schema = 'max_carduner'
input_table_name = f"{catalog}.{schema}.chat_completion_evaluation_dataset"
input_column_name = "outline"
output_table_name = f"{catalog}.{schema}.chat_completion_evaluation_dataset_preds"
output_table_name_ft = f"{catalog}.{schema}.chat_completion_evaluation_dataset_preds_ft"
baseline_endpoint_name = 'llama_v3_2_1b_instruct'
ift_endpoint_name = 'ift-meta-llama-3-1-8b-scexcv'
timeout="300"
max_retries="8"
request_params = '{"max_tokens": 1000, "temperature": 0.1}'
concurrency = "15"
input_num_rows = '1000' # adapt this to your dataset
system_prompt = "You are a marketing professional trusted assistant that helps write rough draft copy into the approved tone and voice. Only focus on the content that is provided by the user, don't add any additional context, just focus on getting it into the approved tone. Do not repeat information, answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer. Given the following outline, write a final copy in our approved tone and voice. Outline:\n"

# Load Configurations from Widgets
config_timeout = int(timeout)
config_max_retries = int(max_retries)

config_prompt = False # we already set the prompt written into the dataset
config_request_params = json.loads(request_params
 ) # Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request

config_concurrecy = int(concurrency)
config_logging_interval = 5

config_input_table = input_table_name
config_input_column = input_column_name
config_input_num_rows = int(input_num_rows)
config_output_table = output_table_name

# COMMAND ----------

from pyspark.sql.functions import col

val_qa_eval = spark.read.table(input_table_name)
display(val_qa_eval)  # noqa

# COMMAND ----------

import os
from mlflow.deployments import get_deploy_client

class ChatClient:
    def __init__(self):
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(config_max_retries)
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(config_timeout)

        self.client = get_deploy_client("databricks")
        # self.endpoint = config_endpoint
        self.request_params = config_request_params
        self.system_prompt = system_prompt

    def predict(self, text, endpoint):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(text)}
            ]

        response = self.client.predict(
            endpoint=endpoint,
            inputs={
                "messages": messages,
                **self.request_params
            }
        )
        return response["choices"][0]["message"]["content"], response["usage"]["total_tokens"]

# COMMAND ----------

def call_batch_udf(config_endpoint, config_output_table, config_input_table=config_input_table,  system_prompt=system_prompt, config_request_params=config_request_params, config_concurrecy=config_concurrecy,config_input_column=config_input_column):
    import time
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from typing import Iterator
    import uuid
    from pyspark.sql.functions import col, cast
    
    @pandas_udf("chat string, total_tokens int, error string")  
    def chat_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        client = ChatClient()

        start_time = time.time()
        total = 0
        for s in iterator:
            chats, total_tokens, errors = [], [], []
            for text in s:
                total += 1
                try:
                    chat, num_token = client.predict(text, config_endpoint)
                    chats.append(chat)
                    total_tokens.append(num_token)
                    errors.append(None)
                except Exception as e:
                    chats.append(None)
                    total_tokens.append(0)
                    errors.append(str(e))

                if total % config_logging_interval == 0:
                    print(f"Processed {total} requests in {time.time() - start_time} seconds")

            yield pd.DataFrame({"chat": chats, "total_tokens": total_tokens, "error": errors})
    
    input_column = f"input_{uuid.uuid4().hex[:4]}"
    output_column = f"output_{uuid.uuid4().hex[:4]}"

    # Step 1: read from the source table
    df = spark.table(config_input_table)
    if config_input_num_rows:
        df = df.limit(config_input_num_rows)

    # Step 2. batch inference
    df = df.repartition(config_concurrecy)  # This is important for performance!!!
    df = df.withColumn(output_column, chat_udf(config_input_column))
    df = (
        df
        .withColumn("resp_chat", col(output_column).getItem("chat"))
        .withColumn("resp_total_tokens", col(output_column).getItem("total_tokens"))
        .withColumn("resp_error", col(output_column).getItem("error"))
        .drop(output_column)
    )
    # Step 3: write to the output table
    df.write.mode("overwrite").saveAsTable(config_output_table)

# COMMAND ----------

call_batch_udf(baseline_endpoint_name, output_table_name)
display(spark.table(config_output_table).limit(10))

# COMMAND ----------

from pyspark.sql.functions import col
eval_df_preds = spark.table(config_output_table)\
                .withColumnRenamed("resp_chat", "pred_baseline")\
                .withColumnRenamed("outline","inputs")\
                .filter(col('resp_error').isNull())\
                .drop("resp_total_tokens", "resp_error").toPandas()
display(eval_df_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC Note: you may want to look into defining your own metric here with a rubric that aligns more closely to your use case: https://docs.databricks.com/en/mlflow/llm-evaluate.html

# COMMAND ----------

import mlflow

# Set the experiment name or path
experiment_name = "/Users/max.carduner@databricks.com/e2e-mosaic-cpt-ift-demo/notebooks/05_evaluate"
mlflow.set_experiment(experiment_name)
run_name = 'llama_v3_2_1b_instruct'
llm_judge = "databricks-meta-llama-3-1-70b-instruct"


with mlflow.start_run(run_name=run_name) as run:
    baseline_results = mlflow.evaluate(
        data=eval_df_preds,
        targets="expected_response",
        predictions="pred_baseline",
        extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
                mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
            ],
        evaluators="default",
    )

baseline_results.metrics

# COMMAND ----------

call_batch_udf(ift_endpoint_name, output_table_name_ft)
display(spark.table(output_table_name_ft).limit(10))

# COMMAND ----------

eval_df_preds_ft = spark.table(output_table_name_ft)\
                .withColumnRenamed("resp_chat", "pred_ft")\
                .withColumnRenamed("outline","inputs")\
                .filter(col('resp_error').isNull())\
                .drop("resp_total_tokens", "resp_error").toPandas()
display(eval_df_preds_ft)

# COMMAND ----------

import mlflow

run_name = 'ft_llama_3_1_8b'
llm_judge = "databricks-meta-llama-3-1-70b-instruct"

with mlflow.start_run(run_name=run_name) as run:
    ft_results = mlflow.evaluate(
        data=eval_df_preds_ft,
        targets="expected_response",
        predictions="pred_ft",
        extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
                mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
            ],
        evaluators="default",
    )

ft_results.metrics
