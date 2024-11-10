# Elevator Pitch Generation App with LLM
Elevator pitch generator built with a fine-tuned version of GPT2, hosted on Streamlit as an app. Check it out [here](https://text-gen-app-iq4ncbimvgwoawcuhzkuha.streamlit.app/).

### Table of Contents
1. [Background](https://github.com/si-rui126/text-gen-app/edit/main/README#background)
2. [Model](https://github.com/si-rui126/text-gen-app/edit/main/README#model)
3. [Model Evaluation](https://github.com/si-rui126/text-gen-app/edit/main/README#model-evaluation)
4. [Deployment](https://github.com/si-rui126/text-gen-app/edit/main/README#deployment)
5. [Conclusion](https://github.com/si-rui126/text-gen-app/edit/main/README#conclusion)

### Background
I wanted practice fine-tuning a LLM and to better understand LLM architecture. Before fine-tuning, I had to think of a use case that would dictate what kind of training data I would use for fine-tuning. It's difficult to reduce a person's experiences, background, and personality into a 30-second pitch, so I thought why not challenge the model to come up with one for me?

### Model
#### Data
I experimented with a variety of datasets to narrow down the model's responses to my desired use case. I ran some prompts with base GPT2 and its responses had the following problems: repetitive and irrelevant. Initially, I used a combination of two datasets:
* dataset of LinkedIn company profiles - to encourage self-marketing language
* dataset of blog posts - to encourage a conversational and narrative language
However, after training GPT2 with these datasets, I still thought the responses were inadequate. Eventually, I resorted to a more brute force-y by scraping websites with examples of elevator pitches for a custom dataset.

#### Training
The project uses gpt-2's API through huggingface's Transformers library. I chose this model because it only has 12 layers, which makes it great to practice on and run many training iterations without being too time intensive. My PC's hardware is also not very strong, so I utilized Low-Rank Adaptation (LoRA) for a more lightweight training process. I initialized the model by calling the AutoModelForCausalLM function since my goal was text generation.
```
llm = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
```
After several iterations of the model and experimenting with different strategies such as Top-k sampling and repetition penalty, I got a model that was able to generate relevant and coherent responses.

Once I was happy with the responses, I compiled the generated output based on common elevator pitch templates. The generated output follows the general template: I am a [position]. During my experience as [previous/current experience]. My goals are [goals]. I believe my [strengths] will...

### Model Evaluation
Comparing the base model to the fine-tuned model with the same prompts: Generate an elevator pitch for a student with experience as a data science intern whose goal is to learn and whose strength is their attention to detail.

Base model response:
```
I am a student. Thank you for your interest in the series," says Professor McGonagall as she leads us through our group of three to see them down one by two, all laughing at her new charms and enchantments before me waving back towards my desk behind Harry Potter with just that little smirk on his face (Harry's got something else he can do too!).
At first I expected this kind oafishly honest explanation from some sort 'good girl', but actually it was really nothing During my experience as a data science intern I've heard quite often from folks that they can't stand the "why", why must be so important. It's only when you really understand how this stuff works and actually talk to people what it is/was before doing experiments, then trying to find an answer for something may improve your decision making but sometimes in some way just doesn´t allow one of those reasons into consideration either or becomes redundant rather than taking on more responsibility like if we were all talking about things directly during our time at work…as opposed perhaps having them stay separate which takes up less space now?
With Project Bias though there are certainly other factors influencing their future project creation plans (and whether through technical constraints), especially regarding human motivation not externalities being involved here: The research field itself depends upon large-scale experimental studies along with good working relationships such – even very small ones depending entirely off scientific curiosity will have far bigger psychological effects because no amount above 5% would help anyone My goals are to learn more about what I need, where and in which places.
```

Fine-tuned model response:
```
I am a student at the University of Colorado System in Fort Collins. During my experience as a data science intern, I always wanted to make it in-house — which is why we now have 10 engineers for every level. My goals are to learn how our people share, read and connect with others. I am open-minded enough to continue striving after learning those things so that all students can become more productive while studying! I believe my analytical abilities will  (for instance) put me in a position of the next world champion (which is not that much fun when people tell you I'm one!).
```
While the fine-tuned model still has a lot of room for improvement, it does a much better job on staying on topic and staying concise. Additionally, the more detailed the prompts, the better the responses.

### Deployment
The fine-tuned model is then integrated into a Streamlit app. The app provides a interface that accepts user input which then is read into the model. Additionally, I utilized Streamlit's cache function, which saves the model as a global resource to improve the app's performance. Streamlit is compatible with GitHub, so deploying as a web app was relatively simple.

### Conclusion
This is my first time fine-tuning a LLM and deploying a Python app, so I learned a lot about LLM architecture, fine-tuning strategies, data preprocessing techniques, and deployment while working on this.


