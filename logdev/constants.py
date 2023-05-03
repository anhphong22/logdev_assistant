# -*- coding:utf-8 -*-
import os
from pathlib import Path
import gradio as gr
from logdev.webui_locale import I18nAuto

i18n = I18nAuto()

CHATGLM_MODEL = None
CHATGLM_TOKENIZER = None
LLAMA_MODEL = None
LLAMA_INFERENCER = None

# ChatGPT configurations
INITIAL_SYSTEM_PROMPT = "You are a helpful assistant."
API_HOST = "api.openai.com"
COMPLETION_URL = "https://api.openai.com/v1/chat/completions"
BALANCE_API_URL = "https://api.openai.com/dashboard/billing/credit_grants"
USAGE_API_URL = "https://api.openai.com/dashboard/billing/usage"
HISTORY_DIR = Path("history")
# HISTORY_DIR = "history"
TEMPLATES_DIR = "templates"

# error message
STANDARD_ERROR_MSG = i18n("☹️An error occurred:")  # Standard prefix for error messages
GENERAL_ERROR_MSG = i18n("An error occurred while getting the dialog, please check the background log")
ERROR_RETRIEVE_MSG = i18n("Please check the network connection, or whether the API-Key is valid.")
CONNECTION_TIMEOUT_MSG = i18n("The connection timed out, unable to get the conversation.")  # Connection timed out
READ_TIMEOUT_MSG = i18n("Read timeout, unable to get conversation.")  # Read timeout
PROXY_ERROR_MSG = i18n("Proxy error, unable to get conversation.")  # Proxy error
SSL_ERROR_PROMPT = i18n("SSL error, unable to get session.")  # SSL error
NO_APIKEY_MSG = i18n(
    "The API key is empty, please check whether the input is correct.")  # The length of the API key is less than 51 digits
NO_INPUT_MSG = i18n("Please enter the dialogue content.")  # No dialogue content is entered
BILLING_NOT_APPLICABLE_MSG = i18n(
    "Billing information is not applicable")  # The billing information returned by the locally running model

TIMEOUT_STREAMING = 60  # Timeout for streaming conversations
TIMEOUT_ALL = 200  # Timeout for non-streaming conversations
ENABLE_STREAMING_OPTION = True  # Whether to enable the check-box to choose whether to display the answer in real time
HIDE_MY_KEY = False  # Set this value to True if you want to hide your API key in UI
CONCURRENT_COUNT = 100  # The number of users allowed to use at the same time

SIM_K = 5
INDEX_QUERY_TEMPERATURE = 1.0

LOGDEV_TITLE = i18n("LogDev Chat 💬")

LOGDEV_DESCRIPTION = i18n("Developed by LogDevHAF Team")

FOOTER = """<div class="versions">{versions}</div>"""

APPEARANCE_SWITCHER = """
<div style="display: flex; justify-content: space-between;">
<span style="margin-top: 4px !important;">""" + i18n("Toggle light and dark theme") + """</span>
<span><label class="apSwitch" for="checkbox">
     <input type="checkbox" id="checkbox">
     <div class="apSlider"></div>
</label></span>
</div>
"""

SUMMARIZE_PROMPT = "Who are you? What did we just talk about?"  # prompt when summarizing the conversation

ONLINE_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "xmchat",
]

LOCAL_MODELS = [
    "chatglm-6b",
    "chatglm-6b-int4",
    "chatglm-6b-int4-qe",
    "StableLM",
    "MOSS",
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]

if os.environ.get('HIDE_LOCAL_MODELS', 'false') == 'true':
    MODELS = ONLINE_MODELS
else:
    MODELS = ONLINE_MODELS + LOCAL_MODELS

DEFAULT_MODEL = 0

os.makedirs("models", exist_ok=True)
os.makedirs("lora", exist_ok=True)
os.makedirs("history", exist_ok=True)
for dir_name in os.listdir("models"):
    if os.path.isdir(os.path.join("models", dir_name)):
        if dir_name not in MODELS:
            MODELS.append(dir_name)

MODEL_TOKEN_LIMIT = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768
}

TOKEN_OFFSET = 1000  # Subtract this value from the token upper limit of the model to get the soft upper limit. After reaching the soft cap, it will automatically try to reduce token usage.
DEFAULT_TOKEN_LIMIT = 3000  # Default token upper limit
REDUCE_TOKEN_FACTOR = 0.5  # Multiply with the model token upper limit to get the target token number. When reducing token occupancy, reduce token occupancy below the target token number.

REPLY_LANGUAGES = [
    "Simplified Chinese",
    "traditional Chinese",
    "English",
    "Japanese",
    "Español",
    "Français",
    "Deutsch",
    "follow problem language (unstable)"
]

WEBSEARCH_PTOMPT_TEMPLATE = """\
Web search results:

{web_results}
Current date: {current_date}

Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
Query: {query}
Reply in {reply_language}
"""

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context_str}
---------------------
Current date: {current_date}.
Using the provided context information, write a comprehensive reply to the given query.
Make sure to cite results using [number] notation after the reference.
If the provided context information refer to multiple subjects with the same name, write separate answers for each subject.
Use prior knowledge only if the given context didn't provide enough information.
Answer the question: {query_str}
Reply in {reply_language}
"""

REFINE_TEMPLATE = """\
The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better
Reply in {reply_language}
If the context isn't useful, return the original answer.
"""

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

small_and_beautiful_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#02C160",
        c100="rgba(2, 193, 96, 0.2)",
        c200="#02C160",
        c300="rgba(2, 193, 96, 0.32)",
        c400="rgba(2, 193, 96, 0.32)",
        c500="rgba(2, 193, 96, 1.0)",
        c600="rgba(2, 193, 96, 1.0)",
        c700="rgba(2, 193, 96, 0.32)",
        c800="rgba(2, 193, 96, 0.32)",
        c900="#02C160",
        c950="#02C160",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f9fafb",
        c100="#f3f4f6",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        c900="#272727",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_background_fill="#06AE56",
    button_primary_background_fill_dark="#06AE56",
    button_primary_background_fill_hover="#07C863",
    button_primary_border_color="#06AE56",
    button_primary_border_color_dark="#06AE56",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F2F2F2",
    button_secondary_background_fill_dark="#2B2B2B",
    button_secondary_text_color="#393939",
    button_secondary_text_color_dark="#FFFFFF",
    # background_fill_primary="#F7F7F7",
    # background_fill_primary_dark="#1F1F1F",
    block_title_text_color="*primary_500",
    block_title_background_fill="*primary_100",
    input_background_fill="#F6F6F6",
)
