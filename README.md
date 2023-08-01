# Building Generative AI Applications with Gradio
https://www.deeplearning.ai/short-courses/
https://learn.deeplearning.ai/huggingface-gradio/lesson/6/chat-with-any-llm

# Install
```bash
# Python 3.10.12
pip install -r requirements.txt
```

# Run
## Summariser / NER
```bash
python gradio_app.py --run_summariser
python gradio_app.py --run_NER
```
## Text <-> Image
```bash
# Text -> Image
clear; python gradio_app.py --run_image_captioning
# Text <- Image
clear; python gradio_app.py --run_image_generation --run_image_generation_cuda
# Text <-> Image
clear; python gradio_app.py --run_image_captioning --run_image_generation --run_image_generation_cuda
```

## Chat bot
```bash
clear; python gradio_app.py --run_ChatBot
```
