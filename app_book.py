import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os

os.environ['API_KEY'] = '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'


st.set_page_config(page_title='Book Recommendation App')

page_bg_img = '''
<style>
h1 {
color: black;
font-weight: bold;
font-size: 48px;  
text-align: center;
margin-top: 50px;  
}

label {
color: black;
font-weight: bold;
font-size: 24px;  
margin-top: 20px;  
}

textarea {
border: 2px solid black;
}

button {
color: black;
font-weight: bold;
font-size: 18px;
border: 2px solid black;
background-color: white;
}

button:hover {
background-color: black;
color: white;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# App title
st.title('Book Recommendation App')

# Text area for input
txt_input = st.text_area('Enter Topics', '', height=20)

def generate_response(input):
    # Generate response from LLM chain
    falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                                repo_id='tiiuae/falcon-7b-instruct',
                                model_kwargs={'temperature': 0.6, 'max_new_tokens': 100})
    template = '''I want to learn {question}, please recommend me some books'''
    prompt = PromptTemplate(template=template, input_variables=['question'])
    falcon_chain = LLMChain(prompt=prompt, llm=falcon_llm, verbose=True)
    output = falcon_chain.run(input)
    return output

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('AI is Thinking...'):
            response = generate_response(txt_input)
            result.append(response)

if len(result):
    st.info(response)
