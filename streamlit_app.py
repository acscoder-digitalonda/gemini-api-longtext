import os
import re
import tiktoken
import google.generativeai as genai
import streamlit as st
import textwrap

from gdocs import gdocs

 
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import group_broken_paragraphs
   
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] 

genai.configure(api_key=GOOGLE_API_KEY)
  
def get_llm(model_name = "gemini-pro"):
    model = genai.GenerativeModel(model_name)
    return lambda text: model.generate_content(text).text
     
elements_to_txt = lambda elements:[str(elem) for elem in elements ]

broken_paragraphs = lambda text: group_broken_paragraphs(text, paragraph_split=re.compile(r"(\s*\n\s*){3}"))   
         
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
    
def llm_prompt(sytem_promt):
    return lambda extra_promt:lambda text: sytem_promt.format(extra_promt=extra_promt,content=text) 
    
if __name__ == "__main__":
     
    sytem_promt = '''You serve as a valuable assistant, adept at enhancing written content and contributing to text improvement.
    {extra_promt}
    Here the content:
    {content}
    '''
   
    
    
    st.title('Gemini API Demonstration')
    st.write('Enter your google document URL ex: https://docs.google.com/document/d/1FKq0wnRDES6PwGKZh0p-DeoZUjUd0VLfwsdxcBlkk/edit.')
    st.write('Make sure you shared that document with acscoder@digitalonda.com or public for everyone can read.')
    
    file_url = st.text_input('Enter your google docs URL')
    extra_prompt = st.text_area('Enter your extra requirement here')
    prompt = llm_prompt(sytem_promt)
    
    llm = get_llm()
    
    if st.button('Submit'):
        with st.spinner('Please wait for the result...'):
            resp = [] 
            chunks = run_doc(file_url)
            for elem in chunks: 
                resp.append(llm(prompt(extra_prompt)(elem.text))) 
            st.write('Here is your result:\n ')
            st.markdown('\n'.join(resp))
             