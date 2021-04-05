## Some Data Science Task

This is an example data science task like the ones companies these days assign to applicants and expect a solution for free. Free stuff better be on Github so here you go.

## How to run (w/o docker)?

First install the requirements (make sure that you have python 3.7 and pipenv beforehand)

`pipenv install -d`

In order to make sure that everything works fine, run the tests

`PYTHONPATH=. pytest tests/test_batch_predict.py`

Now you can invoke the main function (that makes predictions and produces analysis report) using the following

`python main.py -i <PATH_TO_TRANSACTIONS_CSV_FILE> -o <OPTIONAL_PATH_TO_OUTPUT_FOLDER>`

If no output path is passed, the results will be put under `data/output/`

## How to run (with Docker)?

First build the image

`docker build . -t ds_task`

Then run the following to invoke the batch processing script

`docker run -v ${PWD}/data/model:/opt/program/data/model -v <ABSOLUTE_PATH_TO_TRANSACTIONS_CSV_FILE>:/opt/program/infile.csv -v <ABSOLUTE_PATH_TO_THE_OUTPUT_FOLDER>:/opt/program/data/output ds_task python main.py -infile.csv`

## Notebooks

Please check `eda.ipynb` and `lgbmr_quantile.ipynb` under `notebooks/` folder to check the eda and modelling process as well as some visualisations.

### Potential deployment for production

This is a batch processing job. Although it could be deployed as a serverless AWS Lambda function, If the amount of data to be processed high, a sagemaker processing job workflow could be more suitable. In this case, the main.py should be dockerized and be served through an entrypoint. The current folder structure of `data/` is suitable for sagemaker integration.

In any case, serving of the model through an http endpoint should be avoided (e.g. through gunicorn with flask) as it is a batch processing job and the network overhead is unnecessary.

Once the batch job runs, it would upload the results to a key-value store like AWS dynamoDB or redis. Now, the live app can make queries to this database to get the predictions. An important consideration is the update frequency of the existing predictions and the frequency of inclusion of the predictions of new users. Depending on the requirements and specifications agreed with the stakeholders, a running schedule for the batch job should be made. With AWS, it would be possible to schedule a job that extracts the feature data from the data warehouse and upload it to a bucket which then could trigger the above mentioned Lambda or Sagemaker processing job, by which the model is served.