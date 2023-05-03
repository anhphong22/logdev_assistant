from collections import defaultdict
from contextlib import contextmanager
import os
import logging
import sys
import commentjson as json

from . import shared

__all__ = [
    "my_api_key",
    "auth_flag",
    "auth_list",
    "dockerflag",
    "retrieve_proxy",
    "log_level",
    "advance_docs",
    "update_doc_config",
    "multi_api_key",
]

# Add a unified config file to avoid confusion caused by too many files (the lowest priority)
# At the same time, it can also provide config help for subsequent support for custom functions
if os.path.exists("config.json"):
    with open("config.json", "r", encoding='utf-8') as f:
        config = json.load(f)
else:
    config = {}

if os.path.exists("api_key.txt"):
    logging.info("api_key.txt file detected, migration in progress...")
    with open("api_key.txt", "r") as f:
        config["openai_api_key"] = f.read().strip()
    os.rename("api_key.txt", "api_key(deprecated).txt")
    with open("config.json", "w", encoding='utf-8') as f:
        json.dump(config, f, indent=4)

if os.path.exists("auth.json"):
    logging.info("auth.json file detected, migration in progress...")
    auth_list = []
    with open("auth.json", "r", encoding='utf-8') as f:
        auth = json.load(f)
        for _ in auth:
            if auth[_]["username"] and auth[_]["password"]:
                auth_list.append((auth[_]["username"], auth[_]["password"]))
            else:
                logging.error("Please check the username and password in the `auth.json` file!")
                sys.exit(1)
    config["users"] = auth_list
    os.rename("auth.json", "auth(deprecated).json")
    with open("config.json", "w", encoding='utf-8') as f:
        json.dump(config, f, indent=4)

# Dealing with docker if we are running in Docker
dockerflag = config.get("dockerflag", False)
if os.environ.get("dockerrun") == "yes":
    dockerflag = True

# Handle api-key and allowed user list
my_api_key = config.get("openai_api_key", "")
my_api_key = os.environ.get("my_api_key", my_api_key)

# Multi-account mechanism
multi_api_key = config.get("multi_api_key", False)  # Whether to enable the multi-account mechanism
if multi_api_key:
    api_key_list = config.get("api_key_list", [])
    if len(api_key_list) == 0:
        logging.error("The multi-account mode is enabled, but the `api_key_list` is empty, please check `config.json`")
        sys.exit(1)
    shared.state.set_api_key_queue(api_key_list)

auth_list = config.get("users", [])  # is actually a list of users
# Whether to enable the status value of authentication, change to judge the length of auth_list
auth_flag = len(auth_list) > 0

# Process the custom api_host, read the configuration of the environment variable first,
# and automatically assemble it if it exists
api_host = os.environ.get("api_host", config.get("api_host", ""))
if api_host:
    shared.state.set_api_host(api_host)

if dockerflag:
    if my_api_key == "empty":
        logging.error("Please give a api key!")
        sys.exit(1)
    # auth
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if not (isinstance(username, type(None)) or isinstance(password, type(None))):
        auth_list.append((os.environ.get("USERNAME"), os.environ.get("PASSWORD")))
        auth_flag = True


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


# Handle proxy:
http_proxy = config.get("http_proxy", "")
https_proxy = config.get("https_proxy", "")
http_proxy = os.environ.get("HTTP_PROXY", http_proxy)
https_proxy = os.environ.get("HTTPS_PROXY", https_proxy)

# Reset the system variable, do not set the environment variable when it does not need to be set,
# so as not to cause the global agent to report an error
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# Whether to use local embedding
local_embedding = config.get("local_embedding", False)
log_level = config.get("log_level", "INFO")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


@contextmanager
def retrieve_proxy(proxy=None):
    """
    Logic of proxy
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


# Handling advance docs
advance_docs = defaultdict(lambda: defaultdict(dict))
advance_docs.update(config.get("advance_docs", {}))


def update_doc_config(two_column_pdf):
    global advance_docs
    advance_docs["pdf"]["two_column"] = two_column_pdf

    logging.info(f"The updated file parameters are：{advance_docs}")
