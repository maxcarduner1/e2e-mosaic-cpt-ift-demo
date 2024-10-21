# End-to-end Fine-Tuning of LLMs to create customer Tone/Voice chatbot

In this demo, we provide notebooks to aid in the process of building a tone/brand chatbot using the Mosaic AI Model Training stack on Databricks. High level steps are as follows:

1. Prepare data for continued pre-training (CPT) using databricks documentation as a placeholder, but ultimately you should be feeding in past articles, and other pdfs or .txt files you have on brand/tone guidelines
2. Perform CPT (unsupervised)
3. Prepare data for Instruction Fine-Tuning (IFT) by creating synthetic draft of given articles (you will need to provide these articles)
4. Run IFT on CPT model (will need to spin up endpoint after)
5. Evaluate Baseline Model vs. CPT+IFT FT variant
6. Plug into Review App notebooks to spin up a review app with your final model endpoint. Spend time here on modifying the prompt to include a few shot example of what you are looking for and capture any summarized brand guidelines. 