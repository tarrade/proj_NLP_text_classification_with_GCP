import os, sys, pathlib, re, string, unidecode, spacy, bs4, apache_beam as beam

class NLP(beam.DoFn):
        
    def setup(self) -> None:
        self.__spacy = spacy.load('en_core_web_sm')
        self.__spacy.vocab['-pron-'].is_stop = True
        self.__tags = ['javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', 'ios', 'mysql', 'css', 'sql', 'asp.net', 'objective-c', 'ruby-on-rails', '.net', 'c', 'iphone', 'arrays', 'angularjs', 'sql-server', 'ruby', 'json', 'ajax', 'regex', 'r', 'xml', 'asp.net-mvc', 'node.js', 'linux', 'django', 'wpf', 'database', 'xcode', 'vb.net', 'eclipse', 'string', 'swift', 'windows', 'excel', 'wordpress', 'html5', 'spring', 'multithreading', 'facebook', 'image', 'forms', 'git', 'oracle']
        
    def __decode_html(self, input_str: str) -> str:
        try:
            soup = bs4.BeautifulSoup(input_str, 'html.parser')
            text_soup = soup.find_all('p')
            code_soup = soup.find_all('code')
        
            text = [val.get_text(separator=' ') for val in text_soup]
            code = [val.get_text(separator=' ') for val in code_soup]
        
            return ' '.join(text), ' '.join(code)
        except TypeError:
            return '', ''
    
    def __remove_accented_char(self, input_str: str) -> str:
        try:
            return unidecode.unidecode(input_str)
        except AttributeError:
            return ''
    
    def __spacy_cleaning(self, token: str) -> str:
        return not (token.is_stop or token.like_email or token.like_url or token.like_num or token.is_punct)

    def __lemmatization(self, input_str: str) -> str:
        doc = self.__spacy(input_str)
        tokens = [token.lemma_.lower() for token in doc if self.__spacy_cleaning(token)]
        return ' '.join(tokens)
    
    def __remove_punctuation(self, input_str: str) -> str:
        str_punctuation = re.sub('''[!"#$%&\\\\'()*+,-./:;<=>?@[\\]^_`{|}~]+''', ' ', input_str)
        str_white_space = re.sub('''\s+''', ' ', str_punctuation)
        tokens = str_white_space.split(' ')
        output_list = [token for token in tokens if len(token) > 1]
        return ' '.join(output_list)
    
    def __nlp(self, input_str: str) -> str:
        accent = self.__remove_accented_char(input_str)
        lemma = self.__lemmatization(accent)
        punctuation = self.__remove_punctuation(lemma)
        
        return punctuation  
    
    def __get_tags(self, tags: list) ->list:
        return list(set(self.__tags) & set(tags))

    def process(self, element):
        title = self.__nlp(element['title'])
        text, code = self.__decode_html(element['body'])
        text = self.__nlp(text)
        code = self.__nlp(code)
        tags = self.__get_tags(element['tags'])
        
        record = {'id': element['id'],
                 'title': title,
                 'text_body': text,
                 'code_body': code,
                 'tags': tags}
        
        return [record]
    
class CSV(beam.DoFn):
    def process(self, element):
        text_line = str(element['id']) + ', ' + str(element['title'])  + ', ' + str(element['text_body']) \
                     + ', ' + str(element['code_body']) + ', ' + '-'.join(element['tags'])
        return [text_line]