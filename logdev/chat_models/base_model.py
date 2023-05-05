from __future__ import annotations

import json
import logging
import os
import pathlib
import traceback
from enum import Enum

import colorama
import gradio as gr
import tiktoken
import urllib3
from duckduckgo_search import ddg
from llama_index import PromptHelper

from logdev import shared
from logdev.config import local_embedding, retrieve_proxy
from logdev.constants import (
    BILLING_NOT_APPLICABLE_MSG,
    DEFAULT_TOKEN_LIMIT,
    HISTORY_DIR,
    MODEL_TOKEN_LIMIT,
    NO_APIKEY_MSG,
    NO_INPUT_MSG,
    PROMPT_TEMPLATE,
    REDUCE_TOKEN_FACTOR,
    STANDARD_ERROR_MSG,
    TOKEN_OFFSET,
    WEBSEARCH_PTOMPT_TEMPLATE,
    i18n,
)
from logdev.llama_func import construct_index
from logdev.utils import (
    add_details,
    add_source_numbers,
    construct_assistant,
    construct_user,
    count_token,
    get_history_filepath,
    hide_middle_chars,
    new_auto_history_filename,
    replace_today,
    save_file,
)


class ModelType(Enum):
    Unknown = -1
    OpenAI = 0
    ChatGLM = 1
    LLaMA = 2

    @classmethod
    def get_type(cls, model_name: str):
        model_type = None
        model_name_lower = model_name.lower()
        if "gpt" in model_name_lower:
            model_type = ModelType.OpenAI
        elif "chatglm" in model_name_lower:
            model_type = ModelType.ChatGLM
        elif "llama" in model_name_lower or "alpaca" in model_name_lower:
            model_type = ModelType.LLaMA
        else:
            model_type = ModelType.Unknown
        return model_type


class BaseLLMModel:
    def __init__(
        self,
        model_name,
        system_prompt="",
        temperature=1.0,
        top_p=1.0,
        n_choices=1,
        stop=None,
        max_generation_token=None,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias=None,
        user="",
    ) -> None:
        self.history = []
        self.all_token_counts = []
        self.model_name = model_name
        self.model_type = ModelType.get_type(model_name)
        try:
            self.token_upper_limit = MODEL_TOKEN_LIMIT[model_name]
        except KeyError:
            self.token_upper_limit = DEFAULT_TOKEN_LIMIT
        self.interrupted = False
        self.system_prompt = system_prompt
        self.api_key = None
        self.need_api_key = False
        self.single_turn = False

        self.temperature = temperature
        self.top_p = top_p
        self.n_choices = n_choices
        self.stop_sequence = stop
        self.max_generation_token = max_generation_token
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.user_identifier = user

    def get_answer_stream_iter(self):
        """stream predict, need to be implemented
        conversations are stored in self.history, with the most recent question, in OpenAI format
        should return a generator, each time give the next word (str) in the answer
        """
        logging.warning("stream predict not implemented, using at once predict instead")
        response, _ = self.get_answer_at_once()
        yield response

    def get_answer_at_once(self):
        """predict at once, need to be implemented
        conversations are stored in self.history, with the most recent question, in OpenAI format
        Should return:
        the answer (str)
        total token count (int)
        """
        logging.warning("at once predict not implemented, using stream predict instead")
        response_iter = self.get_answer_stream_iter()
        count = 0
        for response in response_iter:
            count += 1
        return response, sum(self.all_token_counts) + count

    def billing_info(self):
        """get billing information, implement if needed"""
        logging.warning("billing info not implemented, using default")
        return BILLING_NOT_APPLICABLE_MSG

    def count_token(self, user_input):
        """get token count from input, implement if needed"""
        # logging.warning("token count not implemented, using default")
        return len(user_input)

    def stream_next_chatbot(self, inputs, chatbot, fake_input=None, display_append=""):
        def get_return_value():
            return chatbot, status_text

        status_text = i18n("Start streaming answers in real time...")
        if fake_input:
            chatbot.append((fake_input, ""))
        else:
            chatbot.append((inputs, ""))

        user_token_count = self.count_token(inputs)
        self.all_token_counts.append(user_token_count)
        logging.debug(f"Count input token: {user_token_count}")

        stream_iter = self.get_answer_stream_iter()

        for partial_text in stream_iter:
            chatbot[-1] = (chatbot[-1][0], partial_text + display_append)
            self.all_token_counts[-1] += 1
            status_text = self.token_message()
            yield get_return_value()
            if self.interrupted:
                self.recover()
                break
        self.history.append(construct_assistant(partial_text))

    def next_chatbot_at_once(self, inputs, chatbot, fake_input=None, display_append=""):
        if fake_input:
            chatbot.append((fake_input, ""))
        else:
            chatbot.append((inputs, ""))
        if fake_input is not None:
            user_token_count = self.count_token(fake_input)
        else:
            user_token_count = self.count_token(inputs)
        self.all_token_counts.append(user_token_count)
        ai_reply, total_token_count = self.get_answer_at_once()
        self.history.append(construct_assistant(ai_reply))
        if fake_input is not None:
            self.history[-2] = construct_user(fake_input)
        chatbot[-1] = (chatbot[-1][0], ai_reply + display_append)
        if fake_input is not None:
            self.all_token_counts[-1] += count_token(construct_assistant(ai_reply))
        else:
            self.all_token_counts[-1] = total_token_count - sum(self.all_token_counts)
        status_text = self.token_message()
        return chatbot, status_text

    def handle_file_upload(self, files, chatbot):
        """if the model accepts multiple model input, implement this function"""
        status = gr.Markdown.update()
        if files:
            construct_index(self.api_key, file_src=files)
            status = "index build completed"
        return gr.Files.update(), chatbot, status

    def prepare_inputs(
        self, real_inputs, use_websearch, files, reply_language, chatbot
    ):
        fake_inputs = None
        display_append = []
        limited_context = False
        fake_inputs = real_inputs
        if files:
            from langchain.chat_models import ChatOpenAI
            from langchain.embeddings.huggingface import HuggingFaceEmbeddings
            from llama_index import (
                GPTSimpleVectorIndex,
                LangchainEmbedding,
                OpenAIEmbedding,
                ServiceContext,
            )
            from llama_index.indices.query.schema import QueryBundle
            from llama_index.indices.vector_store.base_query import (
                GPTVectorStoreIndexQuery,
            )

            limited_context = True
            msg = "Loading index..."
            logging.info(msg)
            # yield chatbot + [(inputs, "")], msg
            index = construct_index(self.api_key, file_src=files)
            assert index is not None, "Failed to get index"
            msg = "Index fetched successfully, generating answer..."
            logging.info(msg)
            if local_embedding or self.model_type != ModelType.OpenAI:
                embed_model = LangchainEmbedding(
                    HuggingFaceEmbeddings(
                        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"
                    )
                )
            else:
                embed_model = OpenAIEmbedding()
            # yield chatbot + [(inputs, "")], msg
            with retrieve_proxy():
                prompt_helper = PromptHelper(
                    max_input_size=4096,
                    num_output=5,
                    max_chunk_overlap=20,
                    chunk_size_limit=600,
                )
                from llama_index import ServiceContext

                service_context = ServiceContext.from_defaults(
                    prompt_helper=prompt_helper, embed_model=embed_model
                )
                query_object = GPTVectorStoreIndexQuery(
                    index.index_struct,
                    service_context=service_context,
                    similarity_top_k=5,
                    vector_store=index._vector_store,
                    docstore=index._docstore,
                    response_synthesizer=None,
                )
                query_bundle = QueryBundle(real_inputs)
                nodes = query_object.retrieve(query_bundle)
            reference_results = [n.node.text for n in nodes]
            reference_results = add_source_numbers(reference_results, use_source=False)
            display_append = add_details(reference_results)
            display_append = "\n\n" + "".join(display_append)
            real_inputs = (
                replace_today(PROMPT_TEMPLATE)
                .replace("{query_str}", real_inputs)
                .replace("{context_str}", "\n\n".join(reference_results))
                .replace("{reply_language}", reply_language)
            )
        elif use_websearch:
            limited_context = True
            search_results = ddg(real_inputs, max_results=5)
            reference_results = []
            for idx, result in enumerate(search_results):
                logging.debug(f"Search result {idx + 1}: {result}")
                domain_name = urllib3.util.parse_url(result["href"]).host
                reference_results.append([result["body"], result["href"]])
                display_append.append(
                    # f"{idx+1}. [{domain_name}]({result['href']})\n"
                    f"<li><a href=\"{result['href']}\" target=\"_blank\">{domain_name}</a></li>\n"
                )
            reference_results = add_source_numbers(reference_results)
            display_append = "<ol>\n\n" + "".join(display_append) + "</ol>"
            real_inputs = (
                replace_today(WEBSEARCH_PTOMPT_TEMPLATE)
                .replace("{query}", real_inputs)
                .replace("{web_results}", "\n\n".join(reference_results))
                .replace("{reply_language}", reply_language)
            )
        else:
            display_append = ""
        return limited_context, fake_inputs, display_append, real_inputs, chatbot

    def predict(
        self,
        inputs,
        chatbot,
        stream=False,
        use_websearch=False,
        files=None,
        reply_language="Vietnamese",
        should_check_token_count=True,
    ):  # repetition_penalty, top_k
        status_text = "Starting to generate answers..."
        logging.info(
            "The input is: "
            + colorama.Fore.BLUE
            + f"{inputs}"
            + colorama.Style.RESET_ALL
        )
        if should_check_token_count:
            yield chatbot + [(inputs, "")], status_text
        if reply_language == "Follow question language (unstable)":
            reply_language = (
                "the same language as the question, such as English, Vietnamese"
            )

        (
            limited_context,
            fake_inputs,
            display_append,
            inputs,
            chatbot,
        ) = self.prepare_inputs(
            real_inputs=inputs,
            use_websearch=use_websearch,
            files=files,
            reply_language=reply_language,
            chatbot=chatbot,
        )
        yield chatbot + [(fake_inputs, "")], status_text

        if (
            self.need_api_key
            and self.api_key is None
            and not shared.state.multi_api_key
        ):
            status_text = STANDARD_ERROR_MSG + NO_APIKEY_MSG
            logging.info(status_text)
            chatbot.append((inputs, ""))
            if len(self.history) == 0:
                self.history.append(construct_user(inputs))
                self.history.append("")
                self.all_token_counts.append(0)
            else:
                self.history[-2] = construct_user(inputs)
            yield chatbot + [(inputs, "")], status_text
            return
        elif len(inputs.strip()) == 0:
            status_text = STANDARD_ERROR_MSG + NO_INPUT_MSG
            logging.info(status_text)
            yield chatbot + [(inputs, "")], status_text
            return

        if self.single_turn:
            self.history = []
            self.all_token_counts = []
        self.history.append(construct_user(inputs))

        try:
            if stream:
                logging.debug("Use streaming")
                iters = self.stream_next_chatbot(
                    inputs,
                    chatbot,
                    fake_input=fake_inputs,
                    display_append=display_append,
                )
                for chatbot, status_text in iters:
                    yield chatbot, status_text
            else:
                logging.debug("Do not use streaming")
                chatbot, status_text = self.next_chatbot_at_once(
                    inputs,
                    chatbot,
                    fake_input=fake_inputs,
                    display_append=display_append,
                )
                yield chatbot, status_text
        except Exception as e:
            traceback.print_exc()
            status_text = STANDARD_ERROR_MSG + str(e)
            yield chatbot, status_text

        if len(self.history) > 1 and self.history[-1]["content"] != inputs:
            logging.info(
                "Answer as："
                + colorama.Fore.BLUE
                + f"{self.history[-1]['content']}"
                + colorama.Style.RESET_ALL
            )

        if limited_context:
            # self.history = self.history[-4:]
            # self.all_token_counts = self.all_token_counts[-2:]
            self.history = []
            self.all_token_counts = []

        max_token = self.token_upper_limit - TOKEN_OFFSET

        if sum(self.all_token_counts) > max_token and should_check_token_count:
            count = 0
            while (
                sum(self.all_token_counts)
                > self.token_upper_limit * REDUCE_TOKEN_FACTOR
                and sum(self.all_token_counts) > 0
            ):
                count += 1
                del self.all_token_counts[0]
                del self.history[:2]
            logging.info(status_text)
            status_text = f"To prevent token overrun, the model forgets {count} rounds of dialogue earlier"
            yield chatbot, status_text

        self.auto_save(chatbot)

    def retry(
        self,
        chatbot,
        stream=False,
        use_websearch=False,
        files=None,
        reply_language="Vietnamese",
    ):
        logging.debug("Retrying……")
        if len(self.history) > 0:
            inputs = self.history[-2]["content"]
            del self.history[-2:]
            self.all_token_counts.pop()
        elif len(chatbot) > 0:
            inputs = chatbot[-1][0]
        else:
            yield chatbot, f"{STANDARD_ERROR_MSG}"
            return

        iters = self.predict(
            inputs,
            chatbot,
            stream=stream,
            use_websearch=use_websearch,
            files=files,
            reply_language=reply_language,
        )
        for x in iters:
            yield x
        logging.debug("Retry Completion")

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False

    def set_token_upper_limit(self, new_upper_limit):
        self.token_upper_limit = new_upper_limit
        print(f"The token upper limit is set to {new_upper_limit}")

    def set_temperature(self, new_temperature):
        self.temperature = new_temperature

    def set_top_p(self, new_top_p):
        self.top_p = new_top_p

    def set_n_choices(self, new_n_choices):
        self.n_choices = new_n_choices

    def set_stop_sequence(self, new_stop_sequence: str):
        new_stop_sequence = new_stop_sequence.split(",")
        self.stop_sequence = new_stop_sequence

    def set_max_tokens(self, new_max_tokens):
        self.max_generation_token = new_max_tokens

    def set_presence_penalty(self, new_presence_penalty):
        self.presence_penalty = new_presence_penalty

    def set_frequency_penalty(self, new_frequency_penalty):
        self.frequency_penalty = new_frequency_penalty

    def set_logit_bias(self, logit_bias):
        logit_bias = logit_bias.split()
        bias_map = {}
        encoding = tiktoken.get_encoding("cl100k_base")
        for line in logit_bias:
            word, bias_amount = line.split(":")
            if word:
                for token in encoding.encode(word):
                    bias_map[token] = float(bias_amount)
        self.logit_bias = bias_map

    def set_user_identifier(self, new_user_identifier):
        self.user_identifier = new_user_identifier

    def set_system_prompt(self, new_system_prompt):
        self.system_prompt = new_system_prompt

    def set_key(self, new_access_key):
        self.api_key = new_access_key.strip()
        msg = i18n("API key changed to") + hide_middle_chars(self.api_key)
        logging.info(msg)
        return self.api_key, msg

    def set_single_turn(self, new_single_turn):
        self.single_turn = new_single_turn

    def reset(self):
        self.history = []
        self.all_token_counts = []
        self.interrupted = False
        pathlib.Path(
            os.path.join(
                HISTORY_DIR,
                self.user_identifier,
                new_auto_history_filename(
                    os.path.join(HISTORY_DIR, self.user_identifier)
                ),
            )
        ).touch()
        return [], self.token_message([0])

    def delete_first_conversation(self):
        if self.history:
            del self.history[:2]
            del self.all_token_counts[0]
        return self.token_message()

    def delete_last_conversation(self, chatbot):
        if len(chatbot) > 0 and STANDARD_ERROR_MSG in chatbot[-1][1]:
            msg = "Due to the error message, only delete the chatbot record"
            chatbot.pop()
            return chatbot, self.history
        if len(self.history) > 0:
            self.history.pop()
            self.history.pop()
        if len(chatbot) > 0:
            msg = "Deleted a set of chatbot conversations"
            chatbot.pop()
        if len(self.all_token_counts) > 0:
            msg = "Deleted token count records for a set of conversations"
            self.all_token_counts.pop()
        msg = "Deleted a set of conversations"
        return chatbot, msg

    def token_message(self, token_lst=None):
        if token_lst is None:
            token_lst = self.all_token_counts
        token_sum = 0
        for i in range(len(token_lst)):
            token_sum += sum(token_lst[: i + 1])
        return (
            i18n("Token count: ")
            + f"{sum(token_lst)}"
            + i18n(", the cumulative consumption of this conversation ")
            + f"{token_sum} tokens"
        )

    def save_chat_history(self, filename, chatbot, user_name):
        if filename == "":
            return
        if not filename.endswith(".json"):
            filename += ".json"
        return save_file(filename, self.system_prompt, self.history, chatbot, user_name)

    def auto_save(self, chatbot):
        history_file_path = get_history_filepath(self.user_identifier)
        save_file(
            history_file_path,
            self.system_prompt,
            self.history,
            chatbot,
            self.user_identifier,
        )

    def export_markdown(self, filename, chatbot, user_name):
        if filename == "":
            return
        if not filename.endswith(".md"):
            filename += ".md"
        return save_file(filename, self.system_prompt, self.history, chatbot, user_name)

    def load_chat_history(self, filename, user_name):
        logging.debug(f"Loading {user_name}'s conversation history ...")
        logging.info(f"filename: {filename}")
        if type(filename) != str and filename is not None:
            filename = filename.name
        try:
            if "/" not in filename:
                history_file_path = os.path.join(HISTORY_DIR, user_name, filename)
            else:
                history_file_path = filename
            with open(history_file_path, "r") as f:
                json_s = json.load(f)
            try:
                if type(json_s["history"][0]) == str:
                    logging.info("History format is legacy, converting...")
                    new_history = []
                    for index, item in enumerate(json_s["history"]):
                        if index % 2 == 0:
                            new_history.append(construct_user(item))
                        else:
                            new_history.append(construct_assistant(item))
                    json_s["history"] = new_history
                    logging.info(new_history)
            except:
                pass
            logging.debug(f"{user_name}'s conversation history loaded")
            self.history = json_s["history"]
            return os.path.basename(filename), json_s["system"], json_s["chatbot"]
        except:
            logging.info(f"No conversation history found {filename}")
            return gr.update(), self.system_prompt, gr.update()

    def auto_load(self):
        if self.user_identifier == "":
            self.reset()
            return self.system_prompt, gr.update()
        history_file_path = get_history_filepath(self.user_identifier)
        filename, system_prompt, chatbot = self.load_chat_history(
            history_file_path, self.user_identifier
        )
        return system_prompt, chatbot

    def like(self):
        """like the last response, implement if needed"""
        return gr.update()

    def dislike(self):
        """dislike the last response, implement if needed"""
        return gr.update()
