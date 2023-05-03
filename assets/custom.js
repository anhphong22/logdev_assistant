// custom javascript here

const MAX_HISTORY_LENGTH = 32;

var key_down_history = [];
var currentIndex = -1;
var user_input_ta;

var gradioContainer = null;
var user_input_ta = null;
var user_input_tb = null;
var userInfoDiv = null;
var appTitleDiv = null;
var chatbot = null;
var apSwitch = null;

var ga = document. getElementsByTagName("gradio-app");
var targetNode = ga[0];
var isInIframe = (window. self !== window. top);

// Did the gradio page load??? Can I move your elements??
function gradioLoaded(mutations) {
     for (var i = 0; i < mutations. length; i++) {
         if (mutations[i]. addedNodes. length) {
             gradioContainer = document.querySelector(".gradio-container");
             user_input_tb = document. getElementById('user_input_tb');
             userInfoDiv = document. getElementById("user_info");
             appTitleDiv = document. getElementById("app_title");
             chatbot = document.querySelector('#chuanhu_chatbot');
             apSwitch = document.querySelector('.apSwitch input[type="checkbox"]');

             if (gradioContainer && apSwitch) { // Is gradioCainter loaded?
                 adjustDarkMode();
             }
             if (user_input_tb) { // Has user_input_tb been loaded yet?
                 selectHistory();
             }
             if (userInfoDiv && appTitleDiv) { // Are userInfoDiv and appTitleDiv loaded?
                 setTimeout(showOrHideUserInfo(), 2000);
             }
             if (chatbot) { // chatbot loaded?
                 setChatbotHeight()
             }
         }
     }
}

function selectHistory() {
     user_input_ta = user_input_tb. querySelector("textarea");
     if (user_input_ta) {
         observer.disconnect(); // stop listening
         // listen for keydown event on textarea
         user_input_ta. addEventListener("keydown", function (event) {
             var value = user_input_ta.value.trim();
             // Determine if the pressed key is an arrow key
             if (event.code === 'ArrowUp' || event.code === 'ArrowDown') {
                 // If the arrow key is pressed, and there is content in the input box, and there is no such content in the history, no operation will be performed
                 if (value && key_down_history. indexOf(value) === -1)
                     return;
                 // For actions that require a response, prevent the default behavior.
                 event. preventDefault();
                 var length = key_down_history. length;
                 if (length === 0) {
                     currentIndex = -1; // If the history record is empty, directly reset the currently selected record
                     return;
                 }
                 if (currentIndex === -1) {
                     currentIndex = length;
                 }
                 if (event. code === 'ArrowUp' && currentIndex > 0) {
                     currentIndex--;
                     user_input_ta.value = key_down_history[currentIndex];
                 } else if (event. code === 'ArrowDown' && currentIndex < length - 1) {
                     currentIndex++;
                     user_input_ta.value = key_down_history[currentIndex];
                 }
                 user_input_ta.selectionStart = user_input_ta.value.length;
                 user_input_ta.selectionEnd = user_input_ta.value.length;
                 const input_event = new InputEvent("input", { bubbles: true, cancelable: true });
                 user_input_ta. dispatchEvent(input_event);
             } else if (event. code === "Enter") {
                 if (value) {
                     currentIndex = -1;
                     if (key_down_history. indexOf(value) === -1) {
                         key_down_history.push(value);
                         if (key_down_history. length > MAX_HISTORY_LENGTH) {
                             key_down_history. shift();
                         }
                     }
                 }
             }
         });
     }
}
function toggleUserInfoVisibility(shouldHide) {
     if (userInfoDiv) {
         if (shouldHide) {
             userInfoDiv.classList.add("hideK");
         } else {
             userInfoDiv. classList. remove("hideK");
         }
     }
}
function showOrHideUserInfo() {
     var sendBtn = document. getElementById("submit_btn");

     // Bind mouse/touch events to show/hide user info
     appTitleDiv. addEventListener("mouseenter", function () {
         toggleUserInfoVisibility(false);
     });
     userInfoDiv. addEventListener("mouseenter", function () {
         toggleUserInfoVisibility(false);
     });
     sendBtn. addEventListener("mouseenter", function () {
         toggleUserInfoVisibility(false);
     });

     appTitleDiv. addEventListener("mouseleave", function () {
         toggleUserInfoVisibility(true);
     });
     userInfoDiv. addEventListener("mouseleave", function () {
         toggleUserInfoVisibility(true);
     });
     sendBtn. addEventListener("mouseleave", function () {
         toggleUserInfoVisibility(true);
     });

     appTitleDiv. ontouchstart = function () {
         toggleUserInfoVisibility(false);
     };
     userInfoDiv.ontouchstart = function () {
         toggleUserInfoVisibility(false);
     };
     sendBtn. ontouchstart = function () {
         toggleUserInfoVisibility(false);
     };

     appTitleDiv. ontouchend = function () {
         setTimeout(function () {
             toggleUserInfoVisibility(true);
         }, 3000);
     };
     userInfoDiv. ontouchend = function () {
         setTimeout(function () {
             toggleUserInfoVisibility(true);
         }, 3000);
     };
     sendBtn. ontouchend = function () {
         setTimeout(function () {
             toggleUserInfoVisibility(true);
         }, 3000); // Delay 1 second to hide user info
     };

     // Hide user info after 2 seconds
     setTimeout(function () {
         toggleUserInfoVisibility(true);
     }, 2000);
}
function toggleDarkMode(isEnabled) {
     if (isEnabled) {
         gradioContainer.classList.add("dark");
         document.body.style.setProperty("background-color", "var(--neutral-950)", "important");
     } else {
         gradioContainer.classList.remove("dark");
         document.body.style.backgroundColor = "";
     }
}
function adjustDarkMode() {
     const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");

     // Set the initial state according to the current color mode
     apSwitch.checked = darkModeQuery.matches;
     toggleDarkMode(darkModeQuery. matches);
     // Listen for color mode changes
     darkModeQuery. addEventListener("change", (e) => {
         apSwitch. checked = e. matches;
         toggleDarkMode(e. matches);
     });
     // apSwitch = document.querySelector('.apSwitch input[type="checkbox"]');
     apSwitch. addEventListener("change", (e) => {
         toggleDarkMode(e. target. checked);
     });
}

function setChatbotHeight() {
     const screenWidth = window. innerWidth;
     const statusDisplay = document. querySelector('#status_display');
     const statusDisplayHeight = statusDisplay ? statusDisplay.offsetHeight : 0;
     const wrap = chatbot. querySelector('. wrap');
     const vh = window. innerHeight * 0.01;
     document.documentElement.style.setProperty('--vh', `${vh}px`);
     if (isInIframe) {
         chatbot.style.height = `700px`;
         wrap.style.maxHeight = `calc(700px - var(--line-sm) * 1rem - 2 * var(--block-label-margin))`
     } else {
         if (screenWidth <= 320) {
             chatbot.style.height = `calc(var(--vh, 1vh) * 100 - ${statusDisplayHeight + 150}px)`;
             wrap.style.maxHeight = `calc(var(--vh, 1vh) * 100 - ${statusDisplayHeight + 150}px - var(--line-sm) * 1rem - 2 * var(--block-label-margin ))`;
         } else if (screenWidth <= 499) {
             chatbot.style.height = `calc(var(--vh, 1vh) * 100 - ${statusDisplayHeight + 100}px)`;
             wrap.style.maxHeight = `calc(var(--vh, 1vh) * 100 - ${statusDisplayHeight + 100}px - var(--line-sm) * 1rem - 2 * var(--block-label-margin ))`;
         } else {
             chatbot.style.height = `calc(var(--vh, 1vh) * 100 - ${statusDisplayHeight + 160}px)`;
             wrap.style.maxHeight = `calc(var(--vh, 1vh) * 100 - ${statusDisplayHeight + 160}px - var(--line-sm) * 1rem - 2 * var(--block-label-margin ))`;
         }
     }
}

// Monitor DOM changes inside the page
var observer = new MutationObserver(function (mutations) {
     gradioLoaded(mutations);
});
observer. observe(targetNode, { childList: true, subtree: true });

// monitor page changes
window. addEventListener("DOMContentLoaded", function () {
     isInIframe = (window. self !== window. top);
});
window.addEventListener('resize', setChatbotHeight);
window.addEventListener('scroll', setChatbotHeight);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", adjustDarkMode);