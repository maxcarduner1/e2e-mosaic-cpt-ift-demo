# Databricks notebook source
# MAGIC %md
# MAGIC # Chat model batch inference tasks using PySpark Pandas UDF
# MAGIC
# MAGIC This notebook is the partner notebook to the **Perform batch inference on a provisioned throughput endpoint** notebook. This notebook and the **Perform batch inference on a provisioned throughput endpoint** notebook must be in the same directory of your workspace for the batch inference workflow to perform successfully.
# MAGIC
# MAGIC The following tasks are accomplished in this notebook: 
# MAGIC
# MAGIC 1. Read data from the input table and input column
# MAGIC 2. Construct the requests and send the requests to a Foundation Model APIs endpoint with some kind of concurrency
# MAGIC 3. Persist input row together with the response data to the output table
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook you need:
# MAGIC
# MAGIC The Databricks Runtime for Machine Learning version 14.3 LTS or above
# MAGIC
# MAGIC Spin up a provisioned throughput endpoint: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis#create-your-provisioned-throughput-endpoint-using-the-ui

# COMMAND ----------

# single node cluster MLR 14.3 16 cores
%pip install -r ../requirements.txt
dbutils.library.restartPython()

# COMMAND ----------

# load in entire documents into pdf

catalog = 'users'
schema = 'max_carduner'
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_data")
dbutils.fs.mkdirs(f"/Volumes/{catalog}/{schema}/raw_data/articles")

# COMMAND ----------

# MAGIC %md
# MAGIC upload article files to raw_data/articles

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from typing import Iterator
import re

# Assuming the volume contains .txt files 
txt_files_volume_path = f"/Volumes/{catalog}/{schema}/raw_data/articles/"

# Read all .txt files from the volume into a dataframe
doc_df = spark.read.text(txt_files_volume_path + "*.txt") \
              .withColumn("path", F.input_file_name())

# remove non alphanumeric characters
def clean(txt):
    txt = re.sub(r"\n", "", txt)
    txt = re.sub(r" ?\.", ".", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s\.,;:!?()\-]", "", txt)
    return txt

# remove any non utf8 encoded characters
def ensure_utf8_encoding(txt):
    # Encode to UTF-8, ignoring errors, then decode back to a string
    txt = clean(txt)
    return txt.encode('utf-8', 'ignore').decode('utf-8')

@pandas_udf("string")
def ensure_utf8_encoding_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for series in batch_iter:
        yield series.apply(ensure_utf8_encoding)

# Assuming doc_df is already defined and has a column named "text"
doc_df = doc_df.withColumn("utf8_text", ensure_utf8_encoding_udf(col("value")))

# display(doc_df)

#Concatenate all lines in each file into a single string, and associate with file name
doc_df = doc_df.groupBy("path") \
    .agg(F.collect_list("utf8_text").alias("all_lines")) \
    .withColumn("text", F.concat_ws("\n", F.col("all_lines"))) \
    .drop("all_lines")

display(doc_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up configuration parameters 

# COMMAND ----------

# DBTITLE 1,configurations
import json

catalog = 'users'
schema = 'max_carduner'
prompt = "Can you provide an outline of the following written content? content:"
input_table_name = f"{catalog}.{schema}.articles" # this shouldn't be splitted docs, entired docs on each line in a df
input_column_name = "text"
output_table_name = f"{catalog}.{schema}.qa_dataset"
endpoint_name = 'meta_llama_v3_1_8b_instruct'
timeout="300"
max_retries="8"
request_params = '{"max_tokens": 1000, "temperature": 0.1}'
concurrency = "15"
input_num_rows = '1000' # adapt this to your dataset
input_column_name = "text"

# Load Configurations from Widgets
config_endpoint = endpoint_name
config_timeout = int(timeout)
config_max_retries = int(max_retries)

config_prompt = prompt
config_request_params = json.loads(request_params
 ) # Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request

config_concurrecy = int(concurrency)
config_logging_interval = 5

config_input_table = input_table_name
config_input_column = input_column_name
config_input_num_rows = int(input_num_rows)
config_output_table = output_table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF for batch inference

# COMMAND ----------

import os
from mlflow.deployments import get_deploy_client

class ChatClient:
    def __init__(self):
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(config_max_retries)
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(config_timeout)

        self.client = get_deploy_client("databricks")
        self.endpoint = config_endpoint
        self.request_params = config_request_params

    def predict(self, text):
        messages = []
        if config_prompt:
            messages.append({"role": "user", "content": config_prompt + str(text)})
        else:
            messages.append({"role": "user", "content": str(text)})

        response = self.client.predict(
            endpoint=self.endpoint,
            inputs={
                "messages": messages,
                **self.request_params
            }
        )
        return response["choices"][0]["message"]["content"], response["usage"]["total_tokens"]

# COMMAND ----------

ChatClient().predict("Databricks is a great platform for building AI applications. It includes a lot of features for building AI applications. One of those is the Review App for human in the loop evaluation. This ensures that you build high quality AI applications by getting feedback before sending them out. Databricks also has the ability to serve provisioned-throughput endpoints of popular SOTA open source models such as the llama family of models")

# COMMAND ----------

import time
import pandas as pd
from pyspark.sql.functions import pandas_udf
from typing import Iterator

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
                chat, num_token = client.predict(text)
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

# COMMAND ----------

# DBTITLE 1,Batch Inference Example
import uuid
from pyspark.sql.functions import col, cast

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

# Check response
training_dataset = spark.read.table(config_output_table)
display(training_dataset.limit(10))

# COMMAND ----------

display(spark.sql(f"select count(*) from {catalog}.{schema}.qa_dataset"))

# COMMAND ----------

# ensure there are no errors, or if there are only a few, will filter out the errors in next step
display(spark.sql(f"select * from {catalog}.{schema}.qa_dataset where resp_error is not null"))

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

# Step 1: Filter data to only those records without a resp_error
df = spark.table(config_output_table)
training_dataset = df.filter(df.resp_error.isNull())\
                      .drop("resp_error","resp_total_tokens")\
                      .selectExpr("resp_chat as outline", "text as expected_response")

#Sean/Lovekush: work on incorporating brand guidelines into this prompt
system_prompt = """You are a marketing professional trusted assistant that helps write rough draft copy into the approved tone and voice. Only focus on the content that is provided by the user, don't add any additional context, just focus on getting it into the approved tone. Do not repeat information, answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer. Given the following outline, write a final copy in our approved tone and voice. Outline:\n"""

@pandas_udf("array<struct<role:string, content:string>>")
def create_conversation(outline: pd.Series, expected_response: pd.Series) -> pd.Series:
    def build_message(o,e):
        user_input = f"Write a final draft based on this outline: {o}."

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": e}]
    return pd.Series([build_message(o,e) for o,e in zip(outline, expected_response)])

training_data, eval_data = training_dataset.randomSplit([0.9, 0.1], seed=42)

training_data.select(create_conversation("outline", "expected_response").alias('messages')).write.mode('overwrite').saveAsTable(f"{catalog}.{schema}.chat_completion_training_dataset")
eval_data.write.mode('overwrite').saveAsTable(f"{catalog}.{schema}.chat_completion_evaluation_dataset")

display(spark.table(f"{catalog}.{schema}.chat_completion_training_dataset"))
