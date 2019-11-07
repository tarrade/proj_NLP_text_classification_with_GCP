import sys
import apache_beam as beam
import bs4
import string
import spacy

class NLPProcessing(beam.DoFn):
    
    def __init__(self):
        self.spacy = None
        
    def start_bundle(self):
        import spacy
        """
        Lazy initialisation of spacy model
        """
        if self.spacy is None:
            self.spacy = spacy.load('en_core_web_sm')
            print('MyLogs: using python {} and beam{}'.format(sys.version, beam.__version__))
        
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
        self.title_array = self.__nlp(element['title'])
        self.body_array = self.__nlp(self.__decode_html(element['body']))
        self.tag_array = self.__split_tags(element['tags'])
        
        print('MyLogs: tittle {}'.format(self.title_array))
        print('MyLogs: body {}'.format(self.body_array))
        print('MyLogs: tags {}'.format(self.tag_array))  
        
        return [{'id': int(element['id']), 
                 'title': ' '.join(self.title_array), 
                 'body': ' '.join(self.body_array), 
                 'tags': [{'value': i} for i in self.tag_array]
                }]