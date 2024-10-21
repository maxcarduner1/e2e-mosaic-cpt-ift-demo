# Databricks notebook source
# MAGIC %md
# MAGIC # Chat model batch inference tasks using [AI Query](https://docs.databricks.com/en/large-language-models/ai-query-batch-inference.html) 
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

# 14.3 MLR LTS
%pip install -r ../requirements.txt
dbutils.library.restartPython()

# COMMAND ----------

# set params
catalog = 'users'
schema = 'max_carduner'
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_data")
dbutils.fs.mkdirs(f"/Volumes/{catalog}/{schema}/raw_data/articles")

prompt = "provide a rough draft of this article, using a casual tone and focusing on just the content, statements, and facts mentioned in the article. Don't say 'Here's a rough draft of the article:' or anything like that before outputting response, just output the rough draft:"
input_table_name = f"{catalog}.{schema}.articles_raw" # this shouldn't be splitted docs, entired docs on each line in a df
input_column_name = "text"
output_table_name = f"{catalog}.{schema}.qa_dataset"
endpoint = 'databricks-meta-llama-3-1-70b-instruct' # swap to a provisioned throughput endpoint for large volume scale after testing with FM API
num_output_tokens = 2000
temperature = 0.1
input_num_rows = '1000' # adapt this to your dataset

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# Create text input widgets for parameters
dbutils.widgets.text("output_table_name", output_table_name)
dbutils.widgets.text("input_column_name", input_column_name)
dbutils.widgets.text("endpoint", endpoint)
dbutils.widgets.text("prompt", prompt)
dbutils.widgets.text("input_table_name", input_table_name)
dbutils.widgets.text("num_output_tokens", str(num_output_tokens))
dbutils.widgets.text("temperature", str(temperature))
dbutils.widgets.text("input_num_rows", input_num_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC upload article files to raw_data/articles (use a few pdfs that aren't just instructions from the intial upload)

# COMMAND ----------

# uncomment below for .txt files
# from pyspark.sql import functions as F
# import pandas as pd
# from pyspark.sql.functions import col, pandas_udf
# from typing import Iterator
# import re

# # Assuming the volume contains .txt files 
# txt_files_volume_path = f"/Volumes/{catalog}/{schema}/raw_data/articles/"

# # Read all .txt files from the volume into a dataframe
# doc_df = spark.read.text(txt_files_volume_path + "*.txt") \
#               .withColumn("path", F.input_file_name())

# # remove non alphanumeric characters
# def clean(txt):
#     txt = re.sub(r"\n", "", txt)
#     txt = re.sub(r" ?\.", ".", txt)
#     txt = re.sub(r"[^a-zA-Z0-9\s\.,;:!?()\-]", "", txt)
#     return txt

# # remove any non utf8 encoded characters
# def ensure_utf8_encoding(txt):
#     # Encode to UTF-8, ignoring errors, then decode back to a string
#     txt = clean(txt)
#     return txt.encode('utf-8', 'ignore').decode('utf-8')

# @pandas_udf("string")
# def ensure_utf8_encoding_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
#     for series in batch_iter:
#         yield series.apply(ensure_utf8_encoding)

# # Assuming doc_df is already defined and has a column named "text"
# doc_df = doc_df.withColumn("utf8_text", ensure_utf8_encoding_udf(col("value")))

# #Concatenate all lines in each file into a single string, and associate with file name
# doc_df = doc_df.groupBy("path") \
#     .agg(F.collect_list("utf8_text").alias("all_lines")) \
#     .withColumn("text", F.concat_ws("\n", F.col("all_lines"))) \
#     .drop("all_lines")

# doc_df.write.mode("overwrite").saveAsTable(input_table_name)

# display(doc_df.limit(5))

# COMMAND ----------

# use below if you want to load in pdfs, otherwise comment out
from finreganalytics.dataprep.dataloading import load_and_clean_data
volume_path = f"/Volumes/{catalog}/{schema}/raw_data/articles/"
doc_df = load_and_clean_data(volume_path)
doc_df = doc_df.filter(doc_df.text.isNotNull() & (doc_df.text != ""))
doc_df.write.mode("overwrite").saveAsTable(input_table_name)
display(doc_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW preprocessed_input AS (
# MAGIC   SELECT
# MAGIC     ${input_column_name},
# MAGIC     SUBSTRING(${input_column_name}, 1, 10000) AS truncated_input_column -- Adjust 10000 based on your needs
# MAGIC   FROM ${input_table_name}
# MAGIC   LIMIT ${input_num_rows}
# MAGIC );
# MAGIC
# MAGIC CREATE OR REPLACE TABLE ${output_table_name} AS (
# MAGIC   SELECT
# MAGIC     ${input_column_name},
# MAGIC     AI_QUERY(
# MAGIC       "${endpoint}",
# MAGIC       CONCAT("${prompt}", truncated_input_column) -- Use the truncated column here
# MAGIC     ) as response
# MAGIC   FROM preprocessed_input
# MAGIC );

# COMMAND ----------

# Check response
training_dataset = spark.read.table(output_table_name)
display(training_dataset.limit(10))

# COMMAND ----------

display(spark.sql(f"select count(*) from {catalog}.{schema}.qa_dataset"))

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

# Step 1: Filter data to only non null responses
df = spark.table(output_table_name)
training_dataset = df.filter(df.response.isNotNull())\
                      .selectExpr("response as draft", "text as expected_response")

#work on incorporating brand guidelines into this prompt
system_prompt = """You are a marketing professional trusted assistant that helps write rough draft copy into the approved tone and voice of our organization. Only focus on the content that is provided by the user, don't add any additional context, just focus on getting it into the approved tone. Do not repeat information, answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer. Given the following draft, write a final copy in our approved tone and voice. draft:\n"""

@pandas_udf("array<struct<role:string, content:string>>")
def create_conversation(draft: pd.Series, expected_response: pd.Series) -> pd.Series:
    def build_message(o,e):
        user_input = f"{o}."

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": e}]
    return pd.Series([build_message(o,e) for o,e in zip(draft, expected_response)])

training_data, eval_data = training_dataset.randomSplit([0.8, 0.2], seed=42)

training_data.select(create_conversation("draft", "expected_response").alias('messages')).write.mode('overwrite').saveAsTable(f"{catalog}.{schema}.chat_completion_training_dataset")
eval_data.write.mode('overwrite').saveAsTable(f"{catalog}.{schema}.chat_completion_evaluation_dataset")

display(spark.table(f"{catalog}.{schema}.chat_completion_training_dataset"))
