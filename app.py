import os
import re
import tiktoken
import google.generativeai as genai

import textwrap
import openai 
from dotenv import load_dotenv
load_dotenv()

from gdocs import gdocs
 
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import group_broken_paragraphs
   
openai.api_key = os.environ.get("OPENAI_API_KEY")   
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
  
def get_llm(model_name = "gemini-pro"):
    model = genai.GenerativeModel(model_name)
    return lambda text: model.generate_content(text).text
     
elements_to_txt = lambda elements:[str(elem) for elem in elements ]

broken_paragraphs = lambda text: group_broken_paragraphs(text, paragraph_split=re.compile(r"(\s*\n\s*){3}"))  
 
def get_openai_llm(model_name = "gpt-4",messages=[]):
    chat = openai.ChatCompletion.create( 
            model=model_name, messages=messages 
        ) 
    reply = chat.choices[0].message.content 
    return reply
    
        
class textChunk:
    text_chunk_length = 4000
    encoding_name = "cl100k_base" 
    def __init__(self,text=""):
        self.text = text
        self.len = self.token_count(text)
            
    def append(self,text):
        self.text += "\n"+text
        self.len += self.token_count(text)
    
    def can_concat(self,text):
        can_concat = self.len + self.token_count(text) < self.text_chunk_length
        return can_concat
         
    def token_count(self,string):
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def __str__(self):
        return self.text

def get_gdoc_id(file_url):
    creds = gdocs.gdoc_creds()
    document_id = gdocs.extract_document_id(file_url)
    return document_id,creds

def run_doc(file_url):
    document_id,creds = get_gdoc_id(file_url)
    chunks = gdocs.read_gdoc_content(creds,document_id)
    #title = gdocs.read_gdoc_title(creds,document_id)

    textChunkList = [textChunk(text="")]
    
    for elem in chunks:
        text = broken_paragraphs( clean( elem, extra_whitespace=True, dashes=True ,bullets=True)  )
        if textChunkList[-1].can_concat(text):
            textChunkList[-1].append(text)
        else:
            textChunkList.append(textChunk(text=text))
    return textChunkList
    
def llm_prompt(sytem_prompt):
    return lambda extra_prompt:lambda text: sytem_prompt.format(extra_prompt=extra_prompt,content=text) 
    
def safe_append(ylist,index, elem):
    try:
        ylist[index] = elem
    except IndexError:
        ylist.append(elem)
        
    return ylist
    
if __name__ == "__main__":
    extra_prompt = ""
    sytem_prompt = '''You serve as a valuable assistant, adept at enhancing written content and contributing to text improvement.
    {extra_prompt}
    Here the content:
    {content}
    '''
    prompt = llm_prompt(sytem_prompt)
    
    llm = get_llm()
     
    file_url = "https://docs.google.com/document/d/1SoGoM9gyVraVwMReC0XCa0DQgLJziS2oz3jdt_Jjcws/edit?usp=sharing" 
    chunks = run_doc(file_url)
    
   
    messages = [ {"role": "system", "content":  
              "You serve as a valuable assistant, adept at enhancing written content and contributing to text improvement."}
                ] 
    for index,elem in enumerate(chunks): 
        messages = safe_append(messages, 3, {"role": "user", "content": elem.text})    
        #reply = get_openai_llm("gpt-4",messages)
        #messages = safe_append(messages, 4, {"role": "assistant", "content": reply})
          
        print(messages)         
                
        print("-----------------------------------------------------------------")
         
    
    