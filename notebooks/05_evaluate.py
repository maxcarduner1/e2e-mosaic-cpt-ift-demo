# Databricks notebook source
# 14.3 MLR LTS
%pip install mlflow[genai,databricks]
dbutils.library.restartPython()

# COMMAND ----------

# set params below
catalog = 'users'
schema = 'max_carduner'
input_table_name = f"{catalog}.{schema}.chat_completion_evaluation_dataset"
input_column_name = "draft"
output_table_name = f"{catalog}.{schema}.chat_completion_evaluation_dataset_preds"
output_table_name_ft = f"{catalog}.{schema}.chat_completion_evaluation_dataset_preds_ft"
baseline_endpoint_name = 'databricks-meta-llama-3-1-70b-instruct' # use PT endpoint for faster inference
ft_endpoint_name = 'ift-meta-llama-3-1-8b-j1lqoh'
input_num_rows = '1000' # adapt this to your dataset
prompt = "You are a marketing professional trusted assistant that helps write rough draft copy into the approved tone and voice. Only focus on the content that is provided by the user, don't add any additional context, just focus on getting it into the approved tone. Do not repeat information, answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer. Given the following draft, write a final copy in our approved tone and voice. Draft:\n"
output_tokens = 5000
temperature = 0.1

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# Create text input widgets for parameters
dbutils.widgets.text("output_table_name", output_table_name)
dbutils.widgets.text("output_table_name_ft", output_table_name_ft)
dbutils.widgets.text("input_column_name", input_column_name)
dbutils.widgets.text("baseline_endpoint", baseline_endpoint_name)
dbutils.widgets.text("ft_endpoint", ft_endpoint_name)
dbutils.widgets.text("prompt", prompt)
dbutils.widgets.text("input_table_name", input_table_name)
dbutils.widgets.text("num_output_tokens", str(output_tokens))
dbutils.widgets.text("temperature", str(temperature))
dbutils.widgets.text("input_num_rows", input_num_rows)

# COMMAND ----------

from pyspark.sql.functions import col

val_qa_eval = spark.read.table(input_table_name)
display(val_qa_eval)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW preprocessed_input AS (
# MAGIC   SELECT
# MAGIC     ${input_column_name},
# MAGIC     expected_response,
# MAGIC     SUBSTRING(${input_column_name}, 1, 10000) AS truncated_input_column -- Adjust 10000 based on your needs
# MAGIC   FROM ${input_table_name}
# MAGIC   LIMIT ${input_num_rows}
# MAGIC );
# MAGIC
# MAGIC CREATE OR REPLACE TABLE ${output_table_name} AS (
# MAGIC   SELECT
# MAGIC     ${input_column_name} as inputs,
# MAGIC     AI_QUERY(
# MAGIC       "${baseline_endpoint}",
# MAGIC       CONCAT("${prompt}", truncated_input_column) -- Use the truncated column here
# MAGIC     ) as response,
# MAGIC     expected_response
# MAGIC   FROM preprocessed_input
# MAGIC ); 

# COMMAND ----------

display(spark.table(output_table_name).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Note: you may want to look into defining your own metric here with a rubric that aligns more closely to your use case: https://docs.databricks.com/en/mlflow/llm-evaluate.html

# COMMAND ----------

import mlflow
from pyspark.sql.functions import col

# read eval table into pandas df for mlflow.evaluate()
eval_df_preds = spark.table(output_table_name)\
                .filter(col('response').isNotNull())\
                .toPandas()

# Set the experiment name or path
experiment_name = "/Users/max.carduner@databricks.com/e2e-mosaic-cpt-ift-demo/notebooks/05_evaluate"
mlflow.set_experiment(experiment_name)
run_name = baseline_endpoint_name
llm_judge = "databricks-meta-llama-3-1-405b-instruct"


with mlflow.start_run(run_name=run_name) as run:
    baseline_results = mlflow.evaluate(
        data=eval_df_preds,
        targets="expected_response",
        predictions="response",
        extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
                mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
            ],
        evaluators="default",
    )

baseline_results.metrics

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ${output_table_name_ft} AS (
# MAGIC   SELECT
# MAGIC     ${input_column_name} as inputs,
# MAGIC     AI_QUERY(
# MAGIC       "${ft_endpoint}",
# MAGIC       CONCAT("${prompt}", truncated_input_column) -- Use the truncated column here
# MAGIC     ) as response,
# MAGIC     expected_response
# MAGIC   FROM preprocessed_input
# MAGIC );

# COMMAND ----------

import mlflow

run_name = ft_endpoint_name

# read eval table into pandas df for mlflow.evaluate()
eval_df_preds_ft = spark.table(output_table_name_ft)\
                .filter(col('response').isNotNull())\
                .toPandas()

with mlflow.start_run(run_name=run_name) as run:
    ft_results = mlflow.evaluate(
        data=eval_df_preds_ft,
        targets="expected_response",
        predictions="response",
        extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
                mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
            ],
        evaluators="default",
    )

ft_results.metrics
