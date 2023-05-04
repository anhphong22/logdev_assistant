import logging
import os
import sys
from collections import defaultdict
from contextlib import contextmanager

import commentjson as json

__all__ = [
    "my_api_key",
    "authflag",
    "auth_list",
    "dockerflag",
    "retrieve_proxy",
    "log_level",
    "advance_docs",
    "update_doc_config",
    "render_latex",
    "usage_limit",
    "multi_api_key",
    "server_name",
    "server_port",
    "share",
    "hide_history_when_not_logged_in",
]

from logdev import constants, shared

if os.path.exists("config.json"):
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

lang_config = config.get("language", "auto")
language = os.environ.get("LANGUAGE", lang_config)

hide_history_when_not_logged_in = config.get("hide_history_when_not_logged_in", False)

if os.path.exists("api_key.txt"):
    logging.info("api_key.txt file detected, migration in progress...")
    with open("api_key.txt", "r") as f:
        config["openai_api_key"] = f.read().strip()
    os.rename("api_key.txt", "api_key(deprecated).txt")
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

if os.path.exists("auth.json"):
    logging.info("auth.json file detected, migration in progress...")
    auth_list = []
    with open("auth.json", "r", encoding="utf-8") as f:
        auth = json.load(f)
        for _ in auth:
            if auth[_]["username"] and auth[_]["password"]:
                auth_list.append((auth[_]["username"], auth[_]["password"]))
            else:
                logging.error(
                    "Please check the username and password in the auth.json file!"
                )
                sys.exit(1)
    config["users"] = auth_list
    os.rename("auth.json", "auth(deprecated).json")
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

# Dealing with docker if we are running in Docker
dockerflag = config.get("dockerflag", False)
if os.environ.get("dockerrun") == "yes":
    dockerflag = True

my_api_key = config.get("openai_api_key", "")
my_api_key = os.environ.get("OPENAI_API_KEY", my_api_key)

xmchat_api_key = config.get("xmchat_api_key", "")
os.environ["XMCHAT_API_KEY"] = xmchat_api_key

render_latex = config.get("render_latex", False)

if render_latex:
    os.environ["RENDER_LATEX"] = "yes"
else:
    os.environ["RENDER_LATEX"] = "no"

usage_limit = os.environ.get("USAGE_LIMIT", config.get("usage_limit", 120))

multi_api_key = config.get("multi_api_key", False)
if multi_api_key:
    api_key_list = config.get("api_key_list", [])
    if len(api_key_list) == 0:
        logging.error(
            "The multi-account mode is enabled, but the api_key_list is empty, please check config.json"
        )
        sys.exit(1)
    shared.state.set_api_key_queue(api_key_list)

auth_list = config.get("users", [])
authflag = len(auth_list) > 0
api_host = os.environ.get("api_host", config.get("api_host", ""))
if api_host:
    shared.state.set_api_host(api_host)


@contextmanager
def retrieve_openai_api(api_key=None):
    old_api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key is None:
        os.environ["OPENAI_API_KEY"] = my_api_key
        yield my_api_key
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        yield api_key
    os.environ["OPENAI_API_KEY"] = old_api_key


log_level = config.get("log_level", "INFO")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

http_proxy = config.get("http_proxy", "")
https_proxy = config.get("https_proxy", "")
http_proxy = os.environ.get("HTTP_PROXY", http_proxy)
https_proxy = os.environ.get("HTTPS_PROXY", https_proxy)

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

local_embedding = config.get("local_embedding", False)


@contextmanager
def retrieve_proxy(proxy=None):
    """
    1. If proxy = NONE, set the environment variable and return the latest set proxy
    2. If proxy! = NONE, update the current proxy configuration, but do not update environment variables
    """
    global http_proxy, https_proxy
    if proxy is not None:
        http_proxy = proxy
        https_proxy = proxy
        yield http_proxy, https_proxy
    else:
        old_var = os.environ["HTTP_PROXY"], os.environ["HTTPS_PROXY"]
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["HTTPS_PROXY"] = https_proxy
        yield http_proxy, https_proxy  # return new proxy

        # return old proxy
        os.environ["HTTP_PROXY"], os.environ["HTTPS_PROXY"] = old_var


advance_docs = defaultdict(lambda: defaultdict(dict))
advance_docs.update(config.get("advance_docs", {}))


def update_doc_config(two_column_pdf):
    global advance_docs
    advance_docs["pdf"]["two_column"] = two_column_pdf

    logging.info(f"The updated documentation parameter is: {advance_docs}")


server_name = config.get("server_name", None)
server_port = config.get("server_port", None)
if server_name is None:
    if dockerflag:
        server_name = "0.0.0.0"
    else:
        server_name = "127.0.0.1"
if server_port is None:
    if dockerflag:
        server_port = 7860

assert (
    server_port is None or type(server_port) == int
), "requires `port` to be set to int type"


default_model = config.get("default_model", "")
try:
    constants.DEFAULT_MODEL = constants.MODELS.index(default_model)
except ValueError:
    pass

share = config.get("share", False)
