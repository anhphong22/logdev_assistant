# -*- coding:utf-8 -*-
import colorama

from logdev import config
from logdev.config import *
from logdev.utils import *
from logdev.overwrites import *
from logdev.chat_models.models import get_model

gr.Chatbot._postprocess_chat_messages = postprocess_chat_messages
gr.Chatbot.postprocess = postprocess
PromptHelper.compact_text_chunks = compact_text_chunks

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()


def create_new_model():
    return get_model(model_name=MODELS[DEFAULT_MODEL], access_key=my_api_key)[0]


with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    user_name = gr.State("")
    promptTemplates = gr.State(load_template(get_template_names(plain=True)[0], mode=2))
    user_question = gr.State("")
    assert type(my_api_key) == str
    user_api_key = gr.State(my_api_key)
    current_model = gr.State(create_new_model)

    topic = gr.State(i18n("Untitled Conversation History"))

    with gr.Row():
        gr.HTML(LOGDEV_TITLE, elem_id="app_title")
        status_display = gr.Markdown(get_geoip(), elem_id="status_display")
    with gr.Row(elem_id="float_display"):
        user_info = gr.Markdown(value="getting user info...", elem_id="user_info")

    with gr.Row().style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="LOGDEV_chatbot").style(height="100%")
            with gr.Row():
                with gr.Column(min_width=225, scale=12):
                    user_input = gr.Textbox(
                        elem_id="user_input_tb",
                        show_label=False, placeholder=i18n("Enter your question..."),
                    ).style(container=False)
                with gr.Column(min_width=42, scale=1):
                    submitBtn = gr.Button(value="", variant="primary", elem_id="submit_btn")
                    cancelBtn = gr.Button(value="", variant="secondary", visible=False, elem_id="cancel_btn")
            with gr.Row():
                emptyBtn = gr.Button(
                    i18n("🧹 New Dialogue"), elem_id="empty_btn"
                )
                retryBtn = gr.Button(i18n("🔄 Regeneration"))
                delFirstBtn = gr.Button(i18n("🗑️ Delete oldest dialog"))
                delLastBtn = gr.Button(i18n("🗑️ Delete latest dialog"))
                with gr.Row(visible=False) as like_dislike_area:
                    with gr.Column(min_width=20, scale=1):
                        likeBtn = gr.Button(i18n("👍"))
                    with gr.Column(min_width=20, scale=1):
                        dislikeBtn = gr.Button(i18n("👎"))

        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label=i18n("Model")):
                    keyTxt = gr.Textbox(
                        show_label=True,
                        placeholder=f"Your API-key...",
                        value=hide_middle_chars(user_api_key.value),
                        type="password",
                        visible=not HIDE_MY_KEY,
                        label="API-Key",
                    )
                    if multi_api_key:
                        usageTxt = gr.Markdown(i18n("Multi-account mode is enabled, no need to enter key, you can start the dialogue directly"),
                                               elem_id="usage_display", elem_classes="insert_block")
                    else:
                        usageTxt = gr.Markdown(i18n("**Send message** or **Submit key** to display credit"), elem_id="usage_display",
                                               elem_classes="insert_block")
                    model_select_dropdown = gr.Dropdown(
                        label=i18n("Select Model"), choices=MODELS, multiselect=False, value=MODELS[DEFAULT_MODEL],
                        interactive=True
                    )
                    lora_select_dropdown = gr.Dropdown(
                        label=i18n("Select LoRA Model"), choices=[], multiselect=False, interactive=True, visible=False
                    )
                    with gr.Row():
                        use_streaming_checkbox = gr.Checkbox(
                            label=i18n("Stream output"), value=True, visible=ENABLE_STREAMING_OPTION
                        )
                        single_turn_checkbox = gr.Checkbox(label=i18n("Single-turn dialogue"), value=False)
                        use_websearch_checkbox = gr.Checkbox(label=i18n("Use online search"), value=False)
                        render_latex_checkbox = gr.Checkbox(
                            label=i18n("Render LaTeX formulas"), value=render_latex, interactive=True,
                            elem_id="render_latex_checkbox"
                        )
                    language_select_dropdown = gr.Dropdown(
                        label=i18n("Select reply language (for search & index)"),
                        choices=REPLY_LANGUAGES,
                        multiselect=False,
                        value=REPLY_LANGUAGES[0],
                    )
                    index_files = gr.Files(label=i18n("Upload"), type="file")
                    two_column = gr.Checkbox(label=i18n("Two-column pdf"), value=advance_docs["pdf"].get("two_column", False))
                    # TODO: formula ocr
                    # formula_ocr = gr.Checkbox(label=i18n("Recognition Formula"), value=advance_docs["pdf"].get("formula_ocr", False))

                with gr.Tab(label="Prompt"):
                    systemPromptTxt = gr.Textbox(
                        show_label=True,
                        placeholder=i18n("Type in System Prompt here..."),
                        label="System prompt",
                        value=INITIAL_SYSTEM_PROMPT,
                        lines=10,
                    ).style(container=False)
                    with gr.Accordion(label=i18n("Load Prompt Template"), open=True):
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=6):
                                    templateFileSelectDropdown = gr.Dropdown(
                                        label=i18n("Select Prompt Template Collection File"),
                                        choices=get_template_names(plain=True),
                                        multiselect=False,
                                        value=get_template_names(plain=True)[0],
                                    ).style(container=False)
                                with gr.Column(scale=1):
                                    templateRefreshBtn = gr.Button(i18n("🔄 Refresh"))
                            with gr.Row():
                                with gr.Column():
                                    templateSelectDropdown = gr.Dropdown(
                                        label=i18n("Load from Prompt Template"),
                                        choices=load_template(
                                            get_template_names(plain=True)[0], mode=1
                                        ),
                                        multiselect=False,
                                    ).style(container=False)

                with gr.Tab(label=i18n("Save/Loa")):
                    with gr.Accordion(label=i18n("Save/Load Dialog History"), open=True):
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=6):
                                    historyFileSelectDropdown = gr.Dropdown(
                                        label=i18n("Load dialog from list"),
                                        choices=get_history_names(plain=True),
                                        multiselect=False
                                    )
                                with gr.Column(scale=1):
                                    historyRefreshBtn = gr.Button(i18n("🔄 Refresh"))
                            with gr.Row():
                                with gr.Column(scale=6):
                                    saveFileName = gr.Textbox(
                                        show_label=True,
                                        placeholder=i18n("Set file name: default is .json, optional is .md"),
                                        label=i18n("Set save file name"),
                                        value=i18n("Dialog History"),
                                    ).style(container=True)
                                with gr.Column(scale=1):
                                    saveHistoryBtn = gr.Button(i18n("💾 Save Dialog"))
                                    exportMarkdownBtn = gr.Button(i18n("📝 Export as Markdown"))
                                    gr.Markdown(i18n("Default save in history folder"))
                            with gr.Row():
                                with gr.Column():
                                    downloadFile = gr.File(interactive=True)

                with gr.Tab(label=i18n("Advanced")):
                    gr.Markdown(i18n("⚠️ Caution: Changes require care. ⚠️\n\nIf unable to use, restore default settings."))
                    gr.HTML(APPEARANCE_SWITCHER, elem_classes="insert_block")
                    with gr.Accordion(i18n("Parameters"), open=False):
                        temperature_slider = gr.Slider(
                            minimum=-0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            interactive=True,
                            label="temperature",
                        )
                        top_p_slider = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            interactive=True,
                            label="top-p",
                        )
                        n_choices_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            interactive=True,
                            label="n choices",
                        )
                        stop_sequence_txt = gr.Textbox(
                            show_label=True,
                            placeholder=i18n("Type in stop token here, separated by comma..."),
                            label="stop",
                            value="",
                            lines=1,
                        )
                        max_context_length_slider = gr.Slider(
                            minimum=1,
                            maximum=32768,
                            value=2000,
                            step=1,
                            interactive=True,
                            label="max context",
                        )
                        max_generation_slider = gr.Slider(
                            minimum=1,
                            maximum=32768,
                            value=1000,
                            step=1,
                            interactive=True,
                            label="max generations",
                        )
                        presence_penalty_slider = gr.Slider(
                            minimum=-2.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.01,
                            interactive=True,
                            label="presence penalty",
                        )
                        frequency_penalty_slider = gr.Slider(
                            minimum=-2.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.01,
                            interactive=True,
                            label="frequency penalty",
                        )
                        logit_bias_txt = gr.Textbox(
                            show_label=True,
                            placeholder=f"word:likelihood",
                            label="logit bias",
                            value="",
                            lines=1,
                        )
                        user_identifier_txt = gr.Textbox(
                            show_label=True,
                            placeholder=i18n("Used to locate abuse"),
                            label=i18n("Username"),
                            value=user_name.value,
                            lines=1,
                        )

                    with gr.Accordion(i18n("Network Settings"), open=False):
                        apihostTxt = gr.Textbox(
                            show_label=True,
                            placeholder=i18n("Type in API-Host here..."),
                            label="API-Host",
                            value=config.api_host or shared.API_HOST,
                            lines=1,
                        )
                        changeAPIURLBtn = gr.Button(i18n("🔄 Switch API Address"))
                        proxyTxt = gr.Textbox(
                            show_label=True,
                            placeholder=i18n("Type in proxy address here..."),
                            label=i18n("Proxy address (example: http://127.0.0.1:10809）"),
                            value="",
                            lines=2,
                        )
                        changeProxyBtn = gr.Button(i18n("🔄 Set Proxy Address"))
                        default_btn = gr.Button(i18n("🔙 Restore Default Settings"))

    gr.Markdown(LOGDEV_DESCRIPTION, elem_id="description")
    gr.HTML(FOOTER.format(versions=versions_html()), elem_id="footer")


    def create_greeting(request: gr.Request):
        if hasattr(request, "username") and request.username:  # is not None or is not ""
            logging.info(f"Get User Name: {request.username}")
            user_info, user_name = gr.Markdown.update(value=f"User: {request.username}"), request.username
        else:
            user_info, user_name = gr.Markdown.update(value=f"", visible=False), ""
        current_model = get_model(model_name=MODELS[DEFAULT_MODEL], access_key=my_api_key)[0]
        current_model.set_user_identifier(user_name)
        return user_info, user_name, current_model, toggle_like_btn_visibility(
            DEFAULT_MODEL), *current_model.auto_load(), get_history_names(False, user_name)


    demo.load(create_greeting, inputs=None,
              outputs=[user_info, user_name, current_model, like_dislike_area, systemPromptTxt, chatbot,
                       historyFileSelectDropdown], api_name="load")
    chatgpt_predict_args = dict(
        fn=predict,
        inputs=[
            current_model,
            user_question,
            chatbot,
            use_streaming_checkbox,
            use_websearch_checkbox,
            index_files,
            language_select_dropdown,
        ],
        outputs=[chatbot, status_display],
        show_progress=True,
    )

    start_outputing_args = dict(
        fn=start_outputting,
        inputs=[],
        outputs=[submitBtn, cancelBtn],
        show_progress=True,
    )

    end_outputing_args = dict(
        fn=end_outputting, inputs=[], outputs=[submitBtn, cancelBtn]
    )

    reset_textbox_args = dict(
        fn=reset_textbox, inputs=[], outputs=[user_input]
    )

    transfer_input_args = dict(
        fn=transfer_input, inputs=[user_input], outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True
    )

    get_usage_args = dict(
        fn=billing_info, inputs=[current_model], outputs=[usageTxt], show_progress=False
    )

    load_history_from_file_args = dict(
        fn=load_chat_history,
        inputs=[current_model, historyFileSelectDropdown, user_name],
        outputs=[saveFileName, systemPromptTxt, chatbot]
    )

    # Chatbot
    cancelBtn.click(interrupt, [current_model], [])

    user_input.submit(**transfer_input_args).then(**chatgpt_predict_args).then(**end_outputing_args)
    user_input.submit(**get_usage_args)

    submitBtn.click(**transfer_input_args).then(**chatgpt_predict_args, api_name="predict").then(**end_outputing_args)
    submitBtn.click(**get_usage_args)

    index_files.change(handle_file_upload, [current_model, index_files, chatbot],
                       [index_files, chatbot, status_display])

    emptyBtn.click(
        reset,
        inputs=[current_model],
        outputs=[chatbot, status_display],
        show_progress=True,
    )

    retryBtn.click(**start_outputing_args).then(
        retry,
        [
            current_model,
            chatbot,
            use_streaming_checkbox,
            use_websearch_checkbox,
            index_files,
            language_select_dropdown,
        ],
        [chatbot, status_display],
        show_progress=True,
    ).then(**end_outputing_args)
    retryBtn.click(**get_usage_args)

    delFirstBtn.click(
        delete_first_conversation,
        [current_model],
        [status_display],
    )

    delLastBtn.click(
        delete_last_conversation,
        [current_model, chatbot],
        [chatbot, status_display],
        show_progress=False
    )

    likeBtn.click(
        like,
        [current_model],
        [status_display],
        show_progress=False
    )

    dislikeBtn.click(
        dislike,
        [current_model],
        [status_display],
        show_progress=False
    )

    two_column.change(update_doc_config, [two_column], None)

    # LLM Models
    keyTxt.change(set_key, [current_model, keyTxt], [user_api_key, status_display], api_name="set_key").then(
        **get_usage_args)
    keyTxt.submit(**get_usage_args)
    single_turn_checkbox.change(set_single_turn, [current_model, single_turn_checkbox], None)
    model_select_dropdown.change(get_model,
                                 [model_select_dropdown, lora_select_dropdown, user_api_key, temperature_slider,
                                  top_p_slider, systemPromptTxt, user_name],
                                 [current_model, status_display, lora_select_dropdown], show_progress=True,
                                 api_name="get_model")
    model_select_dropdown.change(toggle_like_btn_visibility, [model_select_dropdown], [like_dislike_area],
                                 show_progress=False)
    lora_select_dropdown.change(get_model,
                                [model_select_dropdown, lora_select_dropdown, user_api_key, temperature_slider,
                                 top_p_slider, systemPromptTxt, user_name], [current_model, status_display],
                                show_progress=True)

    # Template
    systemPromptTxt.change(set_system_prompt, [current_model, systemPromptTxt], None)
    templateRefreshBtn.click(get_template_names, None, [templateFileSelectDropdown])
    templateFileSelectDropdown.change(
        load_template,
        [templateFileSelectDropdown],
        [promptTemplates, templateSelectDropdown],
        show_progress=True,
    )
    templateSelectDropdown.change(
        get_template_content,
        [promptTemplates, templateSelectDropdown, systemPromptTxt],
        [systemPromptTxt],
        show_progress=True,
    )

    # S&L
    saveHistoryBtn.click(
        save_chat_history,
        [current_model, saveFileName, chatbot, user_name],
        downloadFile,
        show_progress=True,
    )
    saveHistoryBtn.click(get_history_names, [gr.State(False), user_name], [historyFileSelectDropdown])
    exportMarkdownBtn.click(
        export_markdown,
        [current_model, saveFileName, chatbot, user_name],
        downloadFile,
        show_progress=True,
    )
    historyRefreshBtn.click(get_history_names, [gr.State(False), user_name], [historyFileSelectDropdown])
    historyFileSelectDropdown.change(**load_history_from_file_args)
    downloadFile.change(upload_chat_history, [current_model, downloadFile, user_name],
                        [saveFileName, systemPromptTxt, chatbot])

    # Advanced
    max_context_length_slider.change(set_token_upper_limit, [current_model, max_context_length_slider], None)
    temperature_slider.change(set_temperature, [current_model, temperature_slider], None)
    top_p_slider.change(set_top_p, [current_model, top_p_slider], None)
    n_choices_slider.change(set_n_choices, [current_model, n_choices_slider], None)
    stop_sequence_txt.change(set_stop_sequence, [current_model, stop_sequence_txt], None)
    max_generation_slider.change(set_max_tokens, [current_model, max_generation_slider], None)
    presence_penalty_slider.change(set_presence_penalty, [current_model, presence_penalty_slider], None)
    frequency_penalty_slider.change(set_frequency_penalty, [current_model, frequency_penalty_slider], None)
    logit_bias_txt.change(set_logit_bias, [current_model, logit_bias_txt], None)
    user_identifier_txt.change(set_user_identifier, [current_model, user_identifier_txt], None)

    default_btn.click(
        reset_default, [], [apihostTxt, proxyTxt, status_display], show_progress=True
    )
    changeAPIURLBtn.click(
        change_api_host,
        [apihostTxt],
        [status_display],
        show_progress=True,
    )
    changeProxyBtn.click(
        change_proxy,
        [proxyTxt],
        [status_display],
        show_progress=True,
    )

logging.info(
    colorama.Back.GREEN
    + "\nLogDev's warm reminder: Visit http://localhost:7860 to view the interface"
    + colorama.Style.RESET_ALL
)
demo.title = i18n("LogDev Platform 🌌")

if __name__ == "__main__":
    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        auth=auth_list if authflag else None,
        favicon_path="./assets/favicon.ico",
        inbrowser=not dockerflag,
    )
    # demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0", server_port=7860, share=False) # Customizable port
    # demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0", server_port=7860,auth=("Fill in username here", "Fill in password here")) # Username and password can be set
    # demo.queue(concurrency_count=CONCURRENT_COUNT).launch(auth=("Fill in the username here", "Fill in the password here")) # Suitable for Nginx reverse proxy
