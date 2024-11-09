import streamlit as st
from transformers import (AutoTokenizer, AutoModelForCausalLM)
import torch
import os
import re
import argparse
import sys
from dataclasses import dataclass
from time import sleep

############## Getting response from fine-tuned LLM ##############

def get_response(who, exp, goals, qual):
    rp = 1.2 # repetition penalty
    topk = 50 # top-k sampling

    # Hook
    hook_prompt = 'I am a ' + who
    hook_input_ids = tokenizer.encode(hook_prompt, return_tensors='pt')
    hook_am = torch.ones(hook_input_ids.shape, dtype=torch.long)  # Create attention mask
    hook_output = llm.generate(
        hook_input_ids, 
        max_length=100, 
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=hook_am,
        repetition_penalty=rp,
        top_k=topk)
    
    # Current/previous experience
    exp_prompt = 'During my experience as a ' + exp
    exp_input_ids = tokenizer.encode(exp_prompt, return_tensors='pt')
    exp_am = torch.ones(exp_input_ids.shape, dtype=torch.long)  # Create attention mask
    exp_output = llm.generate(
        exp_input_ids, 
        max_length=200, 
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=exp_am,
        repetition_penalty=rp,
        top_k=topk)

    # Goals
    goal_prompt = 'My goals are to ' + goals
    goal_input_ids = tokenizer.encode(goal_prompt, return_tensors='pt')
    goal_am = torch.ones(goal_input_ids.shape, dtype=torch.long)  # Create attention mask
    goal_output = llm.generate(
        goal_input_ids, 
        max_length=200, 
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=goal_am,
        repetition_penalty=rp,
        top_k=topk)
    
    # Strengths
    qual_prompt = 'I believe my ' + qual + ' will '
    qual_input_ids = tokenizer.encode(qual_prompt, return_tensors='pt')
    qual_am = torch.ones(qual_input_ids.shape, dtype=torch.long)  # Create attention mask
    qual_output = llm.generate(
        qual_input_ids, 
        max_length=200, 
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=qual_am,
        repetition_penalty=rp,
        top_k=topk)

    # Preparing final output
    generated_hook = tokenizer.decode(hook_output[0], skip_special_tokens=True)
    generated_exp = tokenizer.decode(exp_output[0], skip_special_tokens=True)
    generated_goal = tokenizer.decode(goal_output[0], skip_special_tokens=True)
    generated_qual = tokenizer.decode(qual_output[0], skip_special_tokens=True)
    final_output = generated_hook + ' ' + generated_exp + ' ' + generated_goal + ' ' + generated_qual
    final_output = re.sub('\n', '', final_output)
    return(final_output)

# page configuration
st.set_page_config(page_title='Generate Elevator Pitch',
                   page_icon='🛗',
                   layout='centered',
                   initial_sidebar_state='collapsed')

# loading model as a cache function
@st.cache_resource
def load_model():
    model_checkpoint = 'final-model'
    model_path = os.path.join('models', model_checkpoint)

    llm = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer
llm, tokenizer = load_model()

############## Start of page ##############
# headers
st.header('🛗 Generate Elevator Pitch')
st.write('Fill out the prompts to generate your own personalized elevator pitch! The more details the better the results :D')

# more columns for additional fields
col1, col2 = st.columns([5,5])
col3, col4 = st.columns([5,5])
with col1:
    in1 = st.text_input('I am a...', placeholder='e.g. student, data scientist')
with col2:
    in2 = st.text_input('Previous/current experience', placeholder='e.g. intern, TA')
with col3:
    in3 = st.text_input('Goals', placeholder='e.g. learn, collaborate, network')
with col4:
    in4 = st.text_input('Strengths', placeholder='e.g. flexibility, attention to detail')

submit = st.button('Generate') # button to generate model response
st.subheader('Output') # header for output

# submit button code
if submit:
    try:
        st.write(get_response(in1, in2, in3, in4))
    except Exception as e:
        st.write(e)
else:
    st.text_area('')

# shameless self-promotion
linkedin_url = "https://www.linkedin.com/in/marian-lu-ba48631a2/"
github_url = "https://github.com/si-rui126"
site_url = "https://si-rui126.github.io/"
st.write("Enjoyed using this generator? Find me at: [LinkedIn](%s)" % linkedin_url + " | [Github](%s)" % github_url + " | [Personal Site](%s)" % site_url)

