import os
import logging
import subprocess
import datetime
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
#import en_core_web_sm
import bs4
import string
import spacy

# export BUCKET_NAME=name_bucket_for_stackoverflow (where you can find /stackoverflow)
# create sub folder on Cloud Storage: gs://{0}/stackoverflow/tmp and gs://{0}/stackoverflow/tmp/staging
# sudo pip3 install -U -r requirements.txt

# get all variables here
os.environ['PROJECT_ID'] =  subprocess.run('gcloud config list project --format "value(core.project)"',
                                             shell=True, check=True,
                                             stdout=subprocess.PIPE).stdout.decode().replace('\n', '').replace('\r', '')

os.environ['REGION'] = subprocess.run('gcloud config get-value compute/region  2> /dev/null',
                                      shell=True, check=True,
                                      stdout=subprocess.PIPE).stdout.decode().replace('\n', '').replace('\r', '')

print(os.environ['PROJECT_ID'])
print(os.environ['BUCKET_NAME'])
print(os.environ['REGION'])

# define query table
def create_query():
    query = """
    SELECT
      id,
      title,
      body,
      tags
    FROM
      `bigquery-public-data.stackoverflow.stackoverflow_posts`
    LIMIT 100
    """

    return query


table_schema = {'fields': [
    {'name': 'id', 'type': 'NUMERIC', 'mode': 'REQUIRED'},
    {'name': 'title', 'type': 'ARRAY', 'mode': 'NULLABLE'},
    {'name': 'body', 'type': 'ARRAY', 'mode': 'NULLABLE'},
    {'name': 'tags', 'type': 'ARRAY', 'mode': 'NULLABLE'},
]}

class CleanText(beam.DoFn):
    import spacy
    import bs4
    
    def __init__(self):
        self.spacy = None
        
    def start_bundle(self):
        """
        Lazy initialisation of spacy model
        """
        if self.spacy is None:
            self.spacy = spacy.load('en_core_web_sm')
        
    def __decode_html(self, input_str: str) -> str:
        self.soup = bs4.BeautifulSoup(input_str, 'html.parser')
        self.output = self.soup.text
        return self.output

    def __nlp(self, input_str: str) -> list:
        self.doc = self.spacy(input_str)
        self.stopwords = list(string.punctuation + string.digits) + ['-pron-']
        self.output = [token.lemma_.lower() for token in self.doc if not token.is_stop 
                  and token.lemma_.lower() not in self.stopwords]
        return self.output

    def __split_tags(self, tags: str) -> list:
        return tags.split('|')

    def process(self, element):
        self.title_array = element['title'] # self.__nlp(element['title'])
        #self.body_decoded = element['tags'] #self.__decode_html(element['tags'])
        self.body_array = element['body']   #self.__nlp(self.body_decoded)
        self.tag_array = element['tags']    #self.__split_tags(element['tags'])
        
        return [{'id': element['id'], 
                 'title': self.title_array, 
                 'body': self.body_array, 
                 'tags': self.tag_array}]


def preprocess():
    """
    Arguments:
        -RUNNER: "DirectRunner" or "DataflowRunner". Specfy to run the pipeline locally or on Google Cloud respectively.
    Side-effects:
        -Creates and executes dataflow pipeline.
        See https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline
    """
    job_name = 'test-stackoverflow' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    project = os.environ['PROJECT_ID']
    region = os.environ['REGION']
    output_dir = "gs://{0}/stackoverflow/".format(os.environ['BUCKET_NAME'])

    # options
    options = PipelineOptions()
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project =  project
    google_cloud_options.job_name =  job_name
    google_cloud_options.region = region
    google_cloud_options.staging_location = os.path.join(output_dir, 'tmp', 'staging')
    google_cloud_options.temp_location = os.path.join(output_dir, 'tmp')
    # done by command line
    #options.view_as(StandardOptions).runner = RUNNER

    # instantantiate Pipeline object using PipelineOptions
    print('Launching Dataflow job {} ... hang on'.format(job_name))
    
    #from setuptools import find_packages
    #print(find_packages())
    #exit(0)

    p = beam.Pipeline(options=options)
    table = p | 'Read from BigQuery' >> beam.io.Read(beam.io.BigQuerySource(
        # query
        query=create_query(),
        # use standard SQL for the above query
        use_standard_sql=True)
        )
    clean_text = table | 'Clean Text' >> beam.ParDo(CleanText())
    #output | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
    #    # The table name is a required argument for the BigQuery
    #    table='test_stackoverflow_beam',
    #    dataset='test',
    #    project=project,
    #    # Here we use the JSON schema read in from a JSON file.
    #    # Specifying the schema allows the API to create the table correctly if it does not yet exist.
    #    schema=table_schema,
    #    # Creates the table in BigQuery if it does not yet exist.
    #    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
    #    # Deletes all data in the BigQuery table before writing.
    #    write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
    #    # not needed, from with clause

    if options.view_as(StandardOptions).runner == 'DataflowRunner':
        print('DataflowRunner')
        p.run()
    else:
        print('Default: DirectRunner')
        result = p.run()
        result.wait_until_finish()
    print('Done')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    print('Starting main process ...')
    preprocess()

# Usage
# python3 test-beam-dataflow.py --runner DataflowRunner
# python3 test-beam-dataflow.py
# python3 test-beam-dataflow.py --runner DataflowRunner --no_use_public_ips --subnetwork https://www.googleapis.com/compute/v1/projects/xxx/regions/europe-west1/subnetworks/yyyy --region=europe-west1 --zone=europe-west1-b
