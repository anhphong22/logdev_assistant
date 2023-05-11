from __future__ import annotations

import datetime
import json
import logging
import os
import platform

import colorama
import commentjson as cjson
import gradio as gr
import requests

from logdev import shared
from logdev.chat_models.base_model import BaseLLMModel, ModelType
from logdev.config import config, retrieve_proxy, usage_limit
from logdev.constants import (
    COMPLETION_URL,
    CONNECTION_TIMEOUT_MSG,
    ERROR_RETRIEVE_MSG,
    GENERAL_ERROR_MSG,
    INITIAL_SYSTEM_PROMPT,
    READ_TIMEOUT_MSG,
    STANDARD_ERROR_MSG,
    TIMEOUT_ALL,
    TIMEOUT_STREAMING,
    i18n,
)
from logdev.utils import (
    construct_system,
    construct_user,
    count_token,
    get_file_names,
    get_last_day_of_month,
)


class OpenAIClient(BaseLLMModel):
    def __init__(
            self,
            model_name,
            api_key,
            system_prompt=INITIAL_SYSTEM_PROMPT,
            temperature=1.0,
            top_p=1.0,
            user_name="",
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user=user_name,
        )
        self.api_key = api_key
        self.need_api_key = True
        self._refresh_header()

    def get_answer_stream_iter(self):
        response = self._get_response(stream=True)
        if response is not None:
            iters = self._decode_chat_response(response)
            partial_text = ""
            for i in iters:
                partial_text += i
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def get_answer_at_once(self):
        response = self._get_response()
        response = json.loads(response.text)
        content = response["choices"][0]["message"]["content"]
        total_token_count = response["usage"]["total_tokens"]
        return content, total_token_count

    def count_token(self, user_input):
        input_token_count = count_token(construct_user(user_input))
        if self.system_prompt is not None and len(self.all_token_counts) == 0:
            system_prompt_token_count = count_token(
                construct_system(self.system_prompt)
            )
            return input_token_count + system_prompt_token_count
        return input_token_count

    def billing_info(self):
        try:
            curr_time = datetime.datetime.now()
            last_day_of_month = get_last_day_of_month(curr_time).strftime("%Y-%m-%d")
            first_day_of_month = curr_time.replace(day=1).strftime("%Y-%m-%d")
            usage_url = f"{shared.state.usage_api_url}?start_date={first_day_of_month}&end_date={last_day_of_month}"
            try:
                usage_data = self._get_billing_data(usage_url)
            except Exception as e:
                logging.error(f"Failed to get API usage: " + str(e))
                return i18n("**Getting API usage failed**")
            rounded_usage = round(usage_data["total_usage"] / 100, 5)
            usage_percent = round(usage_data["total_usage"] / usage_limit, 2)
            return (
                    """\
                    <b>"""
                    + i18n("Amount used this month")
                    + f"""</b>
                <div class="progress-bar">
                    <div class="progress" style="width: {usage_percent}%;">
                        <span class="progress-text">{usage_percent}%</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between;"><span>${rounded_usage}</span><span>${usage_limit}</span></div>
                """
            )
        except requests.exceptions.ConnectTimeout:
            status_text = (
                    STANDARD_ERROR_MSG + CONNECTION_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            )
            return status_text
        except requests.exceptions.ReadTimeout:
            status_text = STANDARD_ERROR_MSG + READ_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            return status_text
        except Exception as e:
            import traceback

            traceback.print_exc()
            logging.error(i18n("Failed to get API usage:") + str(e))
            return STANDARD_ERROR_MSG + ERROR_RETRIEVE_MSG

    def set_token_upper_limit(self, new_upper_limit):
        pass

    @shared.state.switching_api_key
    def _get_response(self, stream=False):
        openai_api_key = self.api_key
        system_prompt = self.system_prompt
        history = self.history
        logging.debug(colorama.Fore.YELLOW + f"{history}" + colorama.Fore.RESET)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }

        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        payload = {
            "model": self.model_name,
            "messages": history,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n_choices,
            "stream": stream,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.max_generation_token is not None:
            payload["max_tokens"] = self.max_generation_token
        if self.stop_sequence is not None:
            payload["stop"] = self.stop_sequence
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if self.user_identifier:
            payload["user"] = self.user_identifier

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # If there is a custom api-host, use the custom host to send the request, otherwise use the default settings to send the request
        if shared.state.completion_url != COMPLETION_URL:
            logging.info(f"Use a custom API URL: {shared.state.completion_url}")

        with retrieve_proxy():
            try:
                response = requests.post(
                    shared.state.completion_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=timeout,
                )
            except:
                return None
        return response

    def _refresh_header(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_billing_data(self, billing_url):
        with retrieve_proxy():
            response = requests.get(
                billing_url,
                headers=self.headers,
                timeout=TIMEOUT_ALL,
            )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

    def _decode_chat_response(self, response):
        error_msg = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode()
                chunk_length = len(chunk)
                try:
                    chunk = json.loads(chunk[6:])
                except json.JSONDecodeError:
                    print(i18n("JSON parsing error, received content: ") + f"{chunk}")
                    error_msg += chunk
                    continue
                if chunk_length > 6 and "delta" in chunk["choices"][0]:
                    if chunk["choices"][0]["finish_reason"] == "stop":
                        break
                    try:
                        yield chunk["choices"][0]["delta"]["content"]
                    except Exception as e:
                        # logging.error(f"Error: {e}")
                        continue
        if error_msg:
            raise Exception(error_msg)

    def set_key(self, new_access_key):
        ret = super().set_key(new_access_key)
        self._refresh_header()
        return ret


class ChatGLM_Client(BaseLLMModel):
    def __init__(self, model_name, user_name="") -> None:
        super().__init__(model_name=model_name, user=user_name)
        import torch
        from transformers import AutoModel, AutoTokenizer

        global CHATGLM_TOKENIZER, CHATGLM_MODEL
        if CHATGLM_TOKENIZER is None or CHATGLM_MODEL is None:
            system_name = platform.system()
            model_path = None
            if os.path.exists("models"):
                model_dirs = os.listdir("models")
                if model_name in model_dirs:
                    model_path = f"models/{model_name}"
            if model_path is not None:
                model_source = model_path
            else:
                model_source = f"THUDM/{model_name}"
            CHATGLM_TOKENIZER = AutoTokenizer.from_pretrained(
                model_source, trust_remote_code=True
            )
            quantified = False
            if "int4" in model_name:
                quantified = True
            model = AutoModel.from_pretrained(model_source, trust_remote_code=True)
            if torch.cuda.is_available():
                # run on CUDA
                logging.info("CUDA is available, using CUDA")
                model = model.half().cuda()
            # There are still some problems with mps acceleration, so don't use it temporarily
            elif system_name == "Darwin" and model_path is not None and not quantified:
                logging.info("Running on macOS, using MPS")
                # running on macOS and model already downloaded
                model = model.half().to("mps")
            else:
                logging.info("GPU is not available, using CPU")
                model = model.float()
            model = model.eval()
            CHATGLM_MODEL = model

    def _get_glm_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        logging.debug(colorama.Fore.YELLOW + f"{history}" + colorama.Fore.RESET)
        assert (
                len(history) % 2 == 0
        ), f"History should be even length. current history is: {history}"
        history = [[history[i], history[i + 1]] for i in range(0, len(history), 2)]
        return history, query

    def get_answer_at_once(self):
        history, query = self._get_glm_style_input()
        response, _ = CHATGLM_MODEL.chat(CHATGLM_TOKENIZER, query, history=history)
        return response, len(response)

    def get_answer_stream_iter(self):
        history, query = self._get_glm_style_input()
        for response, history in CHATGLM_MODEL.stream_chat(
                CHATGLM_TOKENIZER,
                query,
                history,
                max_length=self.token_upper_limit,
                top_p=self.top_p,
                temperature=self.temperature,
        ):
            yield response


class LLaMA_Client(BaseLLMModel):
    def __init__(self, model_name, lora_path=None, user_name="") -> None:
        super().__init__(model_name=model_name, user=user_name)
        from lmflow.args import DatasetArguments, InferencerArguments, ModelArguments
        from lmflow.datasets.dataset import Dataset
        from lmflow.models.auto_model import AutoModel
        from lmflow.pipeline.auto_pipeline import AutoPipeline

        self.max_generation_token = 1000
        self.end_string = "\n\n"
        # We don't need input data
        data_args = DatasetArguments(dataset_path=None)
        self.dataset = Dataset(data_args)
        self.system_prompt = ""

        global LLAMA_MODEL, LLAMA_INFERENCER
        if LLAMA_MODEL is None or LLAMA_INFERENCER is None:
            model_path = None
            if os.path.exists("models"):
                model_dirs = os.listdir("models")
                if model_name in model_dirs:
                    model_path = f"models/{model_name}"
            if model_path is not None:
                model_source = model_path
            else:
                model_source = f"decapoda-research/{model_name}"
            if lora_path is not None:
                lora_path = f"lora/{lora_path}"
            model_args = ModelArguments(
                model_name_or_path=model_source,
                lora_model_path=lora_path,
                model_type=None,
                config_overrides=None,
                config_name=None,
                tokenizer_name=None,
                cache_dir=None,
                use_fast_tokenizer=True,
                model_revision="main",
                use_auth_token=False,
                torch_dtype=None,
                use_lora=False,
                lora_r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                use_ram_optimized_load=True,
            )
            pipeline_args = InferencerArguments(
                local_rank=0,
                random_seed=1,
                deepspeed="configs/ds_config_chatbot.json",
                mixed_precision="bf16",
            )

            with open(pipeline_args.deepspeed, "r") as f:
                ds_config = json.load(f)
            LLAMA_MODEL = AutoModel.get_model(
                model_args,
                tune_strategy="none",
                ds_config=ds_config,
            )
            LLAMA_INFERENCER = AutoPipeline.get_pipeline(
                pipeline_name="inferencer",
                model_args=model_args,
                data_args=data_args,
                pipeline_args=pipeline_args,
            )

    def _get_llama_style_input(self):
        history = []
        instruction = ""
        if self.system_prompt:
            instruction = f"Instruction: {self.system_prompt}\n"
        for x in self.history:
            if x["role"] == "user":
                history.append(f"{instruction}Input: {x['content']}")
            else:
                history.append(f"Output: {x['content']}")
        context = "\n\n".join(history)
        context += "\n\nOutput: "
        return context

    def get_answer_at_once(self):
        context = self._get_llama_style_input()

        input_dataset = self.dataset.from_dict(
            {"type": "text_only", "instances": [{"text": context}]}
        )

        output_dataset = LLAMA_INFERENCER.inference(
            model=LLAMA_MODEL,
            dataset=input_dataset,
            max_new_tokens=self.max_generation_token,
            temperature=self.temperature,
        )

        response = output_dataset.to_dict()["instances"][0]["text"]
        return response, len(response)

    def get_answer_stream_iter(self):
        context = self._get_llama_style_input()
        partial_text = ""
        step = 1
        for _ in range(0, self.max_generation_token, step):
            input_dataset = self.dataset.from_dict(
                {"type": "text_only", "instances": [{"text": context + partial_text}]}
            )
            output_dataset = LLAMA_INFERENCER.inference(
                model=LLAMA_MODEL,
                dataset=input_dataset,
                max_new_tokens=step,
                temperature=self.temperature,
            )
            response = output_dataset.to_dict()["instances"][0]["text"]
            if response == "" or response == self.end_string:
                break
            partial_text += response
            yield partial_text


def get_model(
        model_name,
        lora_model_path=None,
        access_key=None,
        temperature=None,
        top_p=None,
        system_prompt=None,
        user_name="",
) -> BaseLLMModel:  # type: ignore
    msg = i18n("The model is set to:") + f" {model_name}"
    model_type = ModelType.get_type(model_name)
    lora_selector_visibility = False
    lora_choices = []
    dont_change_lora_selector = False
    if model_type != ModelType.OpenAI:
        config.local_embedding = True
    # del current_model.model
    model = None
    try:
        if model_type == ModelType.OpenAI:
            logging.info(f"Loading OpenAI model: {model_name}")
            model = OpenAIClient(
                model_name=model_name,
                api_key=access_key,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                user_name=user_name,
            )
        elif model_type == ModelType.ChatGLM:
            logging.info(f"Loading ChatGLM model: {model_name}")
            model = ChatGLM_Client(model_name, user_name=user_name)
        elif model_type == ModelType.LLaMA and lora_model_path == "":
            msg = f"Now please select the LoRA model for {model_name}"
            logging.info(msg)
            lora_selector_visibility = True
            if os.path.isdir("lora"):
                lora_choices = get_file_names("lora", plain=True, filetypes=[""])
            lora_choices = ["No LoRA"] + lora_choices
        elif model_type == ModelType.LLaMA and lora_model_path != "":
            logging.info(f"Loading LLaMA model: {model_name} + {lora_model_path}")
            dont_change_lora_selector = True
            if lora_model_path == "No LoRA":
                lora_model_path = None
                msg += " + No LoRA"
            else:
                msg += f" + {lora_model_path}"
            model = LLaMA_Client(model_name, lora_model_path, user_name=user_name)
        elif model_type == ModelType.Unknown:
            raise ValueError(f"Unknown model: {model_name}")
        logging.info(msg)
    except Exception as e:
        logging.error(e)
        msg = f"{STANDARD_ERROR_MSG}: {e}"
    if dont_change_lora_selector:
        return model, msg
    else:
        return (
            model,
            msg,
            gr.Dropdown.update(choices=lora_choices, visible=lora_selector_visibility),
        )


if __name__ == "__main__":
    with open("config.json", "r") as f:
        openai_api_key = cjson.load(f)["openai_api_key"]
    # set logging level to debug
    logging.basicConfig(level=logging.DEBUG)
    # client = ModelManager(model_name="gpt-3.5-turbo", access_key=openai_api_key)
    client = get_model(model_name="chatglm-6b-int4")
    chatbot = []
    stream = False
    # Test billing function
    logging.info(colorama.Back.GREEN + "test billing function" + colorama.Back.RESET)
    logging.info(client.billing_info())
    # Test Questions and Answers
    logging.info(
        colorama.Back.GREEN + "Test Questions and Answers" + colorama.Back.RESET
    )
    question = "Is Paris the capital of China?"
    for i in client.predict(inputs=question, chatbot=chatbot, stream=stream):
        logging.info(i)
    logging.info(f"test history after question and answer: {client.history}")
    # test memory
    logging.info(colorama.Back.GREEN + "test memory" + colorama.Back.RESET)
    question = "What question did I just ask you?"
    for i in client.predict(inputs=question, chatbot=chatbot, stream=stream):
        logging.info(i)
    logging.info(f"after memory test history: {client.history}")
    # test retry function
    logging.info(colorama.Back.GREEN + "Test Retry Function" + colorama.Back.RESET)
    for i in client.retry(chatbot=chatbot, stream=stream):
        logging.info(i)
    logging.info(f"retry after history: {client.history}")
    # # Test summary function
    # print(colorama.Back.GREEN + "Test summary function" + colorama.Back.RESET)
    # chatbot, msg = client.reduce_token_size(chatbot=chatbot)
    # print(chatbot, msg)
    # print(f"summarized history: {client.history}")
