# -*- coding:utf-8 -*-
from __future__ import annotations

import csv
import datetime
import html
import json
import logging
import re
import subprocess
import sys
from typing import TYPE_CHECKING, List, Union

import mdtex2html
import pandas as pd
import requests
import tiktoken
from markdown import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from logdev.config import retrieve_proxy
from logdev.constants import *

from . import shared

if TYPE_CHECKING:
    from typing import TypedDict


    class DataframeData(TypedDict):
        headers: List[str]
        data: List[List[str | int | bool]]


def count_token(message):
    encoding = tiktoken.get_encoding("cl100k_base")
    input_str = f"role: {message['role']}, content: {message['content']}"
    length = len(encoding.encode(input_str))
    return length


def markdown_to_html_with_syntax_highlight(md_str):
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)

        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("text", stripall=True)

        formatter = HtmlFormatter()
        highlighted_code = highlight(code, lexer, formatter)

        return f'<pre><code class="{lang}">{highlighted_code}</code></pre>'

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(
                    r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()
            ):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    result += ALREADY_CONVERTED_MARK
    return result


def convert_asis(userinput):
    return (
            f'<p style="white-space:pre-wrap;">{html.escape(userinput)}</p>'
            + ALREADY_CONVERTED_MARK
    )


def detect_converted_mark(userinput):
    if userinput.endswith(ALREADY_CONVERTED_MARK):
        return True
    else:
        return False


def detect_language(code):
    if code.startswith("\n"):
        first_line = ""
    else:
        first_line = code.strip().split("\n", 1)[0]
    language = first_line.lower() if first_line else ""
    code_without_language = code[len(first_line):].lstrip() if first_line else code
    return language, code_without_language


def construct_text(role, text):
    return {"role": role, "content": text}


def construct_user(text):
    return construct_text("user", text)


def construct_system(text):
    return construct_text("system", text)


def construct_assistant(text):
    return construct_text("assistant", text)


def save_file(filename, system, history, chatbot, user_name):
    logging.debug(f"Saving {user_name}'s conversation history")
    os.makedirs(os.path.join(HISTORY_DIR, user_name), exist_ok=True)
    if filename.endswith(".json"):
        json_s = {"system": system, "history": history, "chatbot": chatbot}
        print(json_s)
        with open(os.path.join(HISTORY_DIR, user_name, filename), "w") as f:
            json.dump(json_s, f)
    elif filename.endswith(".md"):
        md_s = f"system: \n- {system} \n"
        for data in history:
            md_s += f"\n{data['role']}: \n- {data['content']} \n"
        with open(os.path.join(HISTORY_DIR, user_name, filename), "w", encoding="utf8") as f:
            f.write(md_s)
    logging.debug(f"{user_name}'s conversation history saved")
    return os.path.join(HISTORY_DIR, user_name, filename)


def get_file_names(directory: Union[str, Path] = None, plain=False, filetypes=None):
    logging.debug(
        f"Get a list of file names, the directory is {directory}, the file type is {filetypes}, whether it is a plain text list {plain}")
    if filetypes is None:
        filetypes = [".json"]
    files = []
    try:
        for file_type in filetypes:
            files += [f for f in os.listdir(directory) if f.endswith(file_type)]
    except FileNotFoundError:
        files = []
    if files is None:
        files = [""]
    logging.debug(f"files are:{files}")
    if plain:
        return files
    else:
        return gr.Dropdown.update(choices=files)


def get_history_names(plain=False, user_name=""):
    logging.debug(f"Get list of history filenames from user {user_name}")
    return get_file_names(os.path.join(HISTORY_DIR, user_name), plain)


def load_template(filename, mode=0):
    logging.debug(
        f"Load the template file {filename}, the mode is {mode} (0 is to return the dictionary and drop-down menu, "
        f"1 is to return the drop-down menu, 2 is to return the dictionary)")
    lines = []
    if filename.endswith(".json"):
        with open(os.path.join(TEMPLATES_DIR, filename), "r", encoding="utf8") as f:
            lines = json.load(f)
        lines = [[i["act"], i["prompt"]] for i in lines]
    else:
        with open(
                os.path.join(TEMPLATES_DIR, filename), "r", encoding="utf8"
        ) as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)
        lines = lines[1:]
    return lines
    # if mode == 1:
    #     return sorted_by_pinyin([row[0] for row in lines])
    # elif mode == 2:
    #     return {row[0]: row[1] for row in lines}
    # else:
    #     choices = sorted_by_pinyin([row[0] for row in lines])
    #     return {row[0]: row[1] for row in lines}, gr.Dropdown.update(
    #         choices=choices
    #     )


def get_template_names(plain=False):
    logging.debug("Get a list of template filenames")
    return get_file_names(TEMPLATES_DIR, plain, filetypes=[".csv", "json"])


def get_template_content(templates, selection, original_system_prompt):
    logging.debug(f"In the application template, the selection is {selection}, and the original system prompt is {original_system_prompt}")
    try:
        return templates[selection]
    except:
        return original_system_prompt


def reset_textbox():
    logging.debug("reset textbox")
    return gr.update(value="")


def reset_default():
    default_host = shared.state.reset_api_host()
    retrieve_proxy("")
    return gr.update(value=default_host), gr.update(value=""), "API-Host and proxy reset"


def change_api_host(host):
    shared.state.set_api_host(host)
    msg = f"API-Host changed to {host}"
    logging.info(msg)
    return msg


def change_proxy(proxy):
    retrieve_proxy(proxy)
    os.environ["HTTPS_PROXY"] = proxy
    msg = f"Proxy changed to {proxy}"
    logging.info(msg)
    return msg


def hide_middle_chars(s):
    if s is None:
        return ""
    if len(s) <= 8:
        return s
    else:
        head = s[:4]
        tail = s[-4:]
        hidden = "*" * (len(s) - 8)
        return head + hidden + tail


def submit_key(key):
    key = key.strip()
    msg = f"API key changed to {hide_middle_chars(key)}"
    logging.info(msg)
    return key, msg


def replace_today(prompt):
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    return prompt.replace("{current_date}", today)


def get_geoip():
    try:
        with retrieve_proxy():
            response = requests.get("https://ipapi.co/json/", timeout=5)
        data = response.json()
    except:
        data = {"error": True, "reason": "Failed to connect to ipapi"}
    if "error" in data.keys():
        logging.warning(f"Unable to obtain IP address information. \n{data}")
        if data["reason"] == "RateLimited":
            return (
                f"Getting IP geolocation failed because the rate limit for detecting IP was reached. Chat functionality may still be available."
            )
        else:
            return f"Failed to obtain IP geolocation. Reason: {data['reason']}. You can still use the chat function."
    else:
        country = data["country_name"]
        if country == "China":
            text = "**Your IP region: China. Please check your proxy settings immediately, using the API in an unsupported region may result in your account being banned. **"
        else:
            text = f"Your IP region: {country}."
        logging.info(text)
        return text


def find_n(lst, max_num):
    n = len(lst)
    total = sum(lst)

    if total < max_num:
        return n

    for i in range(len(lst)):
        if total - lst[i] < max_num:
            return n - i - 1
        total = total - lst[i]
    return 1


def start_outputting():
    logging.debug("Show cancel button, hide send button")
    return gr.Button.update(visible=False), gr.Button.update(visible=True)


def end_outputting():
    return (
        gr.Button.update(visible=True),
        gr.Button.update(visible=False),
    )


def cancel_outputting():
    logging.info("Abort output...")
    shared.state.interrupt()


def transfer_input(inputs):
    # One-time return to reduce latency
    textbox = reset_textbox()
    outputting = start_outputting()
    return (
        inputs,
        gr.update(value=""),
        gr.Button.update(visible=False),
        gr.Button.update(visible=True),
    )


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)
    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                            env=os.environ if custom_env is None else custom_env)
    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)
    return result.stdout.decode(encoding="utf8", errors="ignore")


def versions_html():
    git = os.environ.get('GIT', "git")
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    try:
        commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit_hash = "<none>"
    if commit_hash != "<none>":
        short_commit = commit_hash[0:7]
        commit_info = f"<a style=\"text-decoration:none\" href=\"https://github.com/anhphong22/logdev_assistant/commit/{short_commit}\">{short_commit}</a>"
    else:
        commit_info = "unknown \U0001F615"
    return f"""
Python: <span title="{sys.version}">{python_version}</span>
 • 
Gradio: {gr.__version__}
 • 
Commit: {commit_info}
"""


def add_source_numbers(lst, source_name="Source", use_source=True):
    if use_source:
        return [f'[{idx + 1}]\t "{item[0]}"\n{source_name}: {item[1]}' for idx, item in enumerate(lst)]
    else:
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]


def add_details(lst):
    nodes = []
    for index, txt in enumerate(lst):
        brief = txt[:25].replace("\n", "")
        nodes.append(
            f"<details><summary>{brief}...</summary><p>{txt}</p></details>"
        )
    return nodes


def sheet_to_string(sheet):
    result = []
    for index, row in sheet.iterrows():
        row_string = ""
        for column in sheet.columns:
            row_string += f"{column}: {row[column]}, "
        row_string = row_string.rstrip(", ")
        row_string += "."
        result.append(row_string)
    return result


def excel_to_string(file_path):
    # Read all worksheets in the Excel file
    excel_file = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)

    # Initialize result string
    result = []

    # Loop through each worksheet
    for _, sheet_data in excel_file.items():
        # Process the current worksheet and add to the result string
        result += sheet_to_string(sheet_data)

    return result


def get_last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - datetime.timedelta(days=next_month.day)


def get_model_source(model_name):
    if model_name == "gpt2-medium":
        return "https://huggingface.co/gpt2-medium"
