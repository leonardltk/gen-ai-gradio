import traceback, pdb, pprint
import os
import io
import time
import functools
import base64
from typing import List
import argparse

import gradio as gr
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from termcolor import colored

import torch
from torch import cuda, bfloat16
from PIL import Image

from langchain.llms import HuggingFacePipeline
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, TextStreamer
from diffusers import DiffusionPipeline

def debug_on_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n----- Exception occurred: {e} ----- ")
            traceback.print_exc()
            print(f"----------------------------------------")
            pdb.post_mortem()
    return wrapper

class Summariser():
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        start_time = time.time()

        tokenizer_summary = AutoTokenizer.from_pretrained(self.model_name)
        model_summary = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.pipeline = pipeline("summarization", model=model_summary,  tokenizer=tokenizer_summary,  )
        print_text = f'time taken to load summary model = {time.time() - start_time}'
        print(colored(print_text, 'red'))

        # Test that it works
        start_time = time.time()
        text = ('''The tower is 324 metres (1,063 ft) tall, about the same height
                as an 81-storey building, and the tallest structure in Paris. 
                Its base is square, measuring 125 metres (410 ft) on each side. 
                During its construction, the Eiffel Tower surpassed the Washington 
                Monument to become the tallest man-made structure in the world,
                a title it held for 41 years until the Chrysler Building
                in New York City was finished in 1930. It was the first structure 
                to reach a height of 300 metres. Due to the addition of a broadcasting 
                aerial at the top of the tower in 1957, it is now taller than the 
                Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
                Eiffel Tower is the second tallest free-standing structure in France 
                after the Millau Viaduct.''')
        print(colored(f"text = {text}", 'yellow'))
        text_summarised = self.pipeline(text)
        print(colored(f"text_summarised = {text_summarised}", 'yellow'))
        print_text = f'time taken to run summary inference = {time.time() - start_time}'
        print(colored(print_text, 'red'))

    def summarize(self, input):
        print(colored(input, 'red'))
        output = self.pipeline(input)
        print(colored(output, 'blue'))
        # [{'summary_text': '...'}]
        return output[0]['summary_text']

class NER():
# class NER(BaseModel):
    # model_name: str

    def __init__(self, model_name) -> None:
        # print(f'self.model_name = {self.model_name}')
        # super().__init__(**kwargs)
        # print(f'self.model_name = {self.model_name}')
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.pipeline = pipeline("token-classification", tokenizer=tokenizer, model=model)
        print_text = f'time taken to load NER model = {time.time() - start_time}'
        print(colored(print_text, 'red'))

        # Test that it works
        start_time = time.time()
        text = "My name is Andrew, I'm building DeepLearningAI and I live in California"
        self.pipeline(text)
        print_text = f'time taken to run NER inference = {time.time() - start_time}'
        print(colored(print_text, 'red'))

    def merge_tokens(self, tokens):
        merged_tokens = []
        for idx, token in enumerate(tokens):
            # Initalise list
            if idx == 0:
                merged_tokens.append(token)
                continue

            # If current token continues the entity of the last one, merge them
            TOKEN_IS_INTERMERDIATE = token['entity'].startswith('I-')
            PREV_TOKEN_MATCHES_CURR_TYPE = merged_tokens[-1]['entity'].endswith(token['entity'][2:])
            if merged_tokens and TOKEN_IS_INTERMERDIATE and PREV_TOKEN_MATCHES_CURR_TYPE:
                last_token = merged_tokens[-1]
                last_token['word'] += token['word'].replace('##', '')
                last_token['end'] = token['end']
                last_token['score'] = (last_token['score'] + token['score']) / 2
            else:
                merged_tokens.append(token)

        return merged_tokens

    def ner(self, input):
        """ bert-base-NER
        | Abbreviation  | Description
        | O             | Outside of a named entity
        | B-MIS         | Beginning of a miscellaneous entity right after another miscellaneous entity
        | I-MIS         | Miscellaneous entity
        | B-PER         | Beginning of a person’s name right after another person’s name
        | I-PER         | Person’s name
        | B-ORG         | Beginning of an organization right after another organization
        | I-ORG         | organization
        | B-LOC         | Beginning of a location right after another location
        | I-LOC         | Location
        """
        output = self.pipeline(input)
        print(colored(f'\ninput = {input}', 'blue'))
        for token_dict in output: print(f"\t{token_dict}")
        merged_output = self.merge_tokens(output)
        print(colored(f'merged_output = {merged_output}', 'yellow'))
        """
        [
            {'end': 17,  'entity': 'B-PER',  'index': 4,  'score': 0.9990625,  'start': 11,  'word': 'Andrew'},
            {'end': 36,  'entity': 'B-ORG',  'index': 10,  'score': 0.9927857,  'start': 32,  'word': 'Deep'},
            {'end': 37,  'entity': 'I-ORG',  'index': 11,  'score': 0.99677867,  'start': 36,  'word': '##L'},
            {'end': 40,  'entity': 'I-ORG',  'index': 12,  'score': 0.9954496,  'start': 37,  'word': '##ear'},
            {'end': 44,  'entity': 'I-ORG',  'index': 13,  'score': 0.9959294,  'start': 40,  'word': '##ning'},
            {'end': 45,  'entity': 'I-ORG',  'index': 14,  'score': 0.8917465,  'start': 44,  'word': '##A'},
            {'end': 46,  'entity': 'I-ORG',  'index': 15,  'score': 0.5036115,  'start': 45,  'word': '##I'},
            {'end': 71,  'entity': 'B-LOC',  'index': 20,  'score': 0.99969244,  'start': 61,  'word': 'California'}
        ]
        """
        return {"text": input, "entities": merged_output}

class ImageCaptioning():
    def __init__(self, model_name) -> None:
        self.model_name = model_name

        self.pipeline = None
        self.load_model()

    def load_model(self) -> None:
        # Load pipeline directly
        start_time = time.time()
        self.pipeline = pipeline("image-to-text", model=self.model_name)
        print_text = f'time taken to load ImageCaptioning model = {time.time() - start_time}'
        print(colored(print_text, 'red'))

        # Test that it works
        image_path = 'images/christmas_dog.jpeg'
        print(colored(self.captioner(image_path)), 'yellow')

    def image_to_base64_str(self, pil_image):
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        return str(base64.b64encode(byte_arr).decode('utf-8'))

    def captioner(self, image_path):
        print(colored(f'\nReading from {image_path}', 'blue'))
        # base64_image = self.image_to_base64_str(image)
        result = self.pipeline(image_path)
        pprint.pprint(result)
        return result[0]['generated_text']

class ImageGeneration():
    # https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
    # https://github.com/huggingface/diffusers#popular-tasks--pipelines

    def __init__(self, model_name, use_cuda) -> None:
        self.model_name = model_name

        self.pipeline = None
        self.load_model(use_cuda)

    def load_model(self, use_cuda) -> None:
        # Load pipeline directly
        start_time = time.time()
        self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        if use_cuda:
            self.pipeline.to("cuda")
        print_text = f'time taken to load ImageGeneration model = {time.time() - start_time}'
        print(colored(print_text, 'red'))

    def base64_to_pil(self, img_base64):
        base64_decoded = base64.b64decode(img_base64)
        byte_stream = io.BytesIO(base64_decoded)
        pil_image = Image.open(byte_stream)
        return pil_image

    def generate(self, prompt, negative_prompt, steps, guidance, width, height):
        start_time = time.time()
        print(colored(f'\nprompt = {prompt}', 'blue'))

        params = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height
        }
        print(colored(f'params =', 'red'))
        pprint.pprint(params)

        output = self.pipeline(prompt, **params)
        print(colored(f'output =', 'red'))
        pprint.pprint(output)
        # Potential NSFW content was detected in one or more images.
        # A black image will be returned instead.
        # Try again with a different prompt and/or seed.

        # pil_image = self.base64_to_pil(output.images[0])
        pil_image = output.images[0]
        print(colored(f'time taken to run inference = {time.time() - start_time}', 'yellow'))
        return pil_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple command-line calculator")
    parser.add_argument("--run_summariser", 
                        action="store_true",
                        help="Summarise a long piece of text")
    parser.add_argument("--run_NER", 
                        action="store_true",
                        help="Named Entity Recognizer")
    parser.add_argument("--run_image_captioning", 
                        action="store_true",
                        help="Image to caption generator")
    parser.add_argument("--run_image_generation", 
                        action="store_true",
                        help="Caption to image generator")
    parser.add_argument("--run_image_generation_cuda", 
                        action="store_true",
                        help="whether to accelerate image generation using gpu or not.\
                              Only considered if run_image_generation is true")
    args = parser.parse_args()

    # secret keys
    load_dotenv(find_dotenv()) # read local .env file
    hf_api_key = os.environ['HF_API_KEY']

    # ----- Summarise -----
    if args.run_summariser:
        model_name_summary = "sshleifer/distilbart-cnn-12-6" # https://huggingface.co/sshleifer/distilbart-cnn-12-6
        summariser = Summariser(model_name_summary)
    # ----- NER -----
    if args.run_NER:
        model_name_NER = "dslim/bert-base-NER" # https://huggingface.co/dslim/bert-base-NER
        ner = NER(model_name=model_name_NER)
    # ----- Image Captioning -----
    if args.run_image_captioning:
        model_name_ImageCaption = "Salesforce/blip-image-captioning-base" # https://huggingface.co/Salesforce/blip-image-captioning-base
        image_captioning = ImageCaptioning(model_name_ImageCaption)
    # ----- Image Generation -----
    if args.run_image_generation:
        model_name_ImageCaption = "runwayml/stable-diffusion-v1-5" # https://huggingface.co/runwayml/stable-diffusion-v1-5
        image_generation = ImageGeneration(model_name_ImageCaption, use_cuda=args.run_image_generation_cuda)


    # Gradio Interfaces
    if args.run_summariser:
        demo = gr.Interface(fn=summariser.summarize, 
                        inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                        outputs=[gr.Textbox(label="Result", lines=3)],
                        title="Text summarization with distilbart-cnn",
                        description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                        )
    elif args.run_NER:
        demo = gr.Interface(fn=ner.ner,
                            inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                            outputs=[gr.HighlightedText(label="Text with entities")],
                            title="NER with dslim/bert-base-NER",
                            description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                            allow_flagging="never",
                            examples=["My name is Andrew and I live in California", 
                                    "My name is Poli and work at HuggingFace"])
    elif args.run_image_captioning and args.run_image_generation:
        # image to image
        def caption_and_generate(image, negative_prompt,steps,guidance,width,height):
            caption = image_captioning.captioner(image)
            image = image_generation.generate(caption, negative_prompt,steps,guidance,width,height)
            return [caption, image]

        with gr.Blocks() as demo:
            # Advanced options
            with gr.Accordion("Advanced options", 
                              open=False): # Let's hide the advanced options!
                negative_prompt = gr.Textbox(label="Negative prompt")
                with gr.Row():
                    with gr.Column():
                        steps = gr.Slider(label="Inference Steps",
                                          minimum=1, maximum=100, value=25,
                                          info="In many steps will the denoiser denoise the image?")
                        guidance = gr.Slider(label="Guidance Scale",
                                             minimum=1, maximum=20, value=7,
                                             info="Controls how much the text prompt influences the result")
                    with gr.Column():
                        width = gr.Slider(label="Width",
                                          minimum=64, maximum=512, step=64, value=512)
                        height = gr.Slider(label="Height",
                                           minimum=64, maximum=512, step=64, value=512)
            # Image to caption to image
            with gr.Row():
                gr.Markdown("# Image to caption to image")
                # image to caption
                image_upload = gr.Image(label="Your first image",type="pil")
                btn_caption = gr.Button("Generate caption")
                caption = gr.Textbox(label="Generated caption")
                # caption to image
                btn_image = gr.Button("Generate image")
                image_output = gr.Image(label="Generated Image")
                # buttons
                btn_caption.click(fn=image_captioning.captioner,
                                  inputs=[image_upload],
                                  outputs=[caption])
                btn_image.click(fn=image_generation.generate,
                                inputs=[caption, negative_prompt,steps,guidance,width,height],
                                outputs=[image_output])
            # Image to image
            with gr.Row():
                gr.Markdown("# Image to image")
                image_upload = gr.Image(label="Your first image",type="pil")
                btn_all = gr.Button("Caption and generate")
                caption = gr.Textbox(label="Generated caption")
                image_output = gr.Image(label="Generated Image")

                btn_all.click(fn=caption_and_generate,
                              inputs=[image_upload, negative_prompt,steps,guidance,width,height],
                              outputs=[caption, image_output])
    elif args.run_image_captioning:
        demo = gr.Interface(fn=image_captioning.captioner,
                            inputs=[gr.Image(label="Upload image", type="pil")],
                            outputs=[gr.Textbox(label="Caption")],
                            title="Image Captioning with BLIP",
                            description="Caption any image using the BLIP model",
                            allow_flagging="never",
                            examples=["images/christmas_dog.jpeg"])
    elif args.run_image_generation:
        # Fixed default parameters run
        demo = gr.Interface(fn=image_generation.generate,
                            inputs=[gr.Textbox(label="Your prompt")],
                            outputs=[gr.Image(label="Result")],
                            title="Image Generation with Stable Diffusion",
                            description="Generate any image with Stable Diffusion",
                            allow_flagging="never",
                            examples=["the spirit of a tamagotchi wandering in the city of Vienna",
                                      "a mecha robot in a favela"],
                            )
        # Sliders to enable parameter tuning
        demo = gr.Interface(fn=image_generation.generate,
                    inputs=[
                        gr.Textbox(label="Your prompt",
                                   value='a dog in a park'),
                        gr.Textbox(label="Negative prompt",
                                   value='low quality'),
                        gr.Slider(label="Inference Steps",
                                  minimum=1, maximum=100, value=1,
                                  info="In how many steps will the denoiser denoise the image?"),
                        gr.Slider(label="Guidance Scale",
                                  minimum=1, maximum=20, value=7, 
                                  info="Controls how much the text prompt influences the result"),
                        gr.Slider(label="Width",
                                  minimum=64, maximum=512, step=64, value=512),
                        gr.Slider(label="Height",
                                  minimum=64, maximum=512, step=64, value=512),
                    ],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    )
        # Gradio Blocks
        # """
        with gr.Blocks() as demo:
            gr.Markdown("# Image Generation with Stable Diffusion")
            prompt = gr.Textbox(label="Your prompt")
            with gr.Row():
                with gr.Column():
                    negative_prompt = gr.Textbox(label="Negative prompt",
                                                 value="low quality"
                                                 )
                    steps = gr.Slider(label="Inference Steps", 
                                      minimum=1, maximum=100, value=25,
                                      info="In many steps will the denoiser denoise the image?"
                                      )
                    guidance = gr.Slider(label="Guidance Scale", 
                                         minimum=1, maximum=20, value=7,
                                         info="Controls how much the text prompt influences the result"
                                         )
                    width = gr.Slider(label="Width",
                                      minimum=64, maximum=512, 
                                      step=64, value=512
                                      )
                    height = gr.Slider(label="Height",
                                       minimum=64, maximum=512, 
                                       step=64, value=512
                                       )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output = gr.Image(label="Result")

            btn.click(fn=image_generation.generate, 
                      inputs=[prompt,
                              negative_prompt,
                              steps,
                              guidance,
                              width,
                              height], 
                      outputs=[output]
                      )
        # """
        # """
        with gr.Blocks() as demo:
            gr.Markdown("# Image Generation with Stable Diffusion")
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(label="Your prompt") # Give prompt some real estate
                with gr.Column(scale=1, min_width=50):
                    btn = gr.Button("Submit") # Submit button side by side!
            with gr.Accordion("Advanced options", 
                              open=False): # Let's hide the advanced options!
                negative_prompt = gr.Textbox(label="Negative prompt")
                with gr.Row():
                    with gr.Column():
                        steps = gr.Slider(label="Inference Steps",
                                            minimum=1, maximum=100, value=25,
                                            info="In many steps will the denoiser denoise the image?")
                        guidance = gr.Slider(label="Guidance Scale",
                                                minimum=1, maximum=20, value=7,
                                                info="Controls how much the text prompt influences the result")
                    with gr.Column():
                        width = gr.Slider(label="Width",
                                            minimum=64, maximum=512, step=64, value=512)
                        height = gr.Slider(label="Height",
                                            minimum=64, maximum=512, step=64, value=512)
            output = gr.Image(label="Result") # Move the output up too
                    
            btn.click(fn=image_generation.generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])
        # """

    # Gradio Launch
    demo.launch(server_port=5004,
                share=False,
                show_error=True,
                show_tips=True,
                )

