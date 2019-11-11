import os, sys, pathlib, re, string, unidecode, spacy, bs4, apache_beam as beam

class NLP(beam.DoFn):
    
    def __init__(self):
        pass
        
    def start_bundle(self):
        self.__spacy = spacy.load('en_core_web_sm')
        self.__spacy.vocab['-pron-'].is_stop = True
        
    def __decode_html(self, input_str: str) -> str:
        soup = bs4.BeautifulSoup(input_str, 'html.parser')
        text_soup = soup.find_all('p')
        code_soup = soup.find_all('code')
        
        text = [val.get_text(separator=' ') for val in text_soup]
        code = [val.get_text(separator=' ') for val in code_soup]
        
        return ' '.join(text), ' '.join(code)
    
    def __remove_accented_char(self, input_str: str) -> str:
        try:
            return unidecode.unidecode(input_str)
        except AttributeError:
            return ''
    
    def __spacy_cleaning(self, token: str) -> str:
        return (not (token.is_stop or token.like_email or token.like_url or token.like_num or token.is_punct) and 
                len(token) > 1)

    def __lemmatization(self, input_str: str) -> str:
        doc = self.__spacy(input_str)
        tokens = [token.lemma_.lower() for token in doc if self.__spacy_cleaning(token)]
        return ' '.join(tokens)
    
    def __remove_punctuation(self, input_str):
        str_punctuation = re.sub('''[!"#$%&\\'()*+,-./:;<=>?@[\\]^_`{|}~]+''', ' ', input_str)
        str_white_space = re.sub('''\s+''', ' ', str_punctuation)
        return str_white_space
    
    def __split_tags(self, tags: str) -> list:
        return tags.split('|')
    
    def __nlp(self, input_str: str) -> str:
        accent = self.__remove_accented_char(input_str)
        lemma = self.__lemmatization(accent)
        punctuation = self.__remove_punctuation(lemma)
        
        return punctuation

    def process(self, element):
        title = self.__nlp(element['title'])
        text, code = self.__decode_html(element['body'])
        text = self.__nlp(text)
        code = self.__nlp(code)
        tags = self.__split_tags(element['tags'])
        
        record = {'id': element['id'],
                 'title': title,
                 'text_body': text,
                 'code_body': code,
                 'tags': [{'value': val} for val in tags]}
        
        return [record]