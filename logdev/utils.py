# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING, List
import re
import html
import tiktoken
from markdown import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

from .constants import ALREADY_CONVERTED_MARK

if TYPE_CHECKING:
    from typing import TypedDict


    class DataframeData(TypedDict):
        headers: List[str]
        data: List[List[str | int | bool]]


def predict(current_model, *args):
    iters = current_model.predict(*args)
    for i in iters:
        yield i


def billing_info(current_model):
    return current_model.billing_info()


def set_key(current_model, *args):
    return current_model.set_key(*args)


def load_chat_history(current_model, *args):
    return current_model.load_chat_history(*args)


def interrupt(current_model, *args):
    return current_model.interrupt(*args)


def reset(current_model, *args):
    return current_model.reset(*args)


def retry(current_model, *args):
    iters = current_model.retry(*args)
    for i in iters:
        yield i


def delete_first_conversation(current_model, *args):
    return current_model.delete_first_conversation(*args)


def delete_last_conversation(current_model, *args):
    return current_model.delete_last_conversation(*args)


def set_system_prompt(current_model, *args):
    return current_model.set_system_prompt(*args)


def save_chat_history(current_model, *args):
    return current_model.save_chat_history(*args)


def export_markdown(current_model, *args):
    return current_model.export_markdown(*args)


def load_chat_history(current_model, *args):
    return current_model.load_chat_history(*args)


def upload_chat_history(current_model, *args):
    return current_model.load_chat_history(*args)


def set_token_upper_limit(current_model, *args):
    return current_model.set_token_upper_limit(*args)


def set_temperature(current_model, *args):
    current_model.set_temperature(*args)


def set_top_p(current_model, *args):
    current_model.set_top_p(*args)


def set_n_choices(current_model, *args):
    current_model.set_n_choices(*args)


def set_stop_sequence(current_model, *args):
    current_model.set_stop_sequence(*args)


def set_max_tokens(current_model, *args):
    current_model.set_max_tokens(*args)


def set_presence_penalty(current_model, *args):
    current_model.set_presence_penalty(*args)


def set_frequency_penalty(current_model, *args):
    current_model.set_frequency_penalty(*args)


def set_logit_bias(current_model, *args):
    current_model.set_logit_bias(*args)


def set_user_identifier(current_model, *args):
    current_model.set_user_identifier(*args)


def set_single_turn(current_model, *args):
    current_model.set_single_turn(*args)


def handle_file_upload(current_model, *args):
    return current_model.handle_file_upload(*args)


def like(current_model, *args):
    return current_model.like(*args)


def dislike(current_model, *args):
    return current_model.dislike(*args)


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
    raw = f'<div class="raw-message hideM">{html.escape(md_text)}</div>'
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            result.append(markdown(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    output = f'<div class="md-message">{result}</div>'
    output += raw
    output += ALREADY_CONVERTED_MARK
    return output


def convert_asis(userinput):
    return (
            f'<p style="white-space:pre-wrap;">{html.escape(userinput)}</p>'
            + ALREADY_CONVERTED_MARK
    )


def detect_converted_mark(userinput):
    try:
        if userinput.endswith(ALREADY_CONVERTED_MARK):
            return True
        else:
            return False
    except:
        return True


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
