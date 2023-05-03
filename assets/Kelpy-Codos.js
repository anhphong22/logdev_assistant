(function () {
     'use strict';

     function addCopyButton(pre) {
         var code = pre. querySelector('code');
         if (!code) {
             return; // If no <code> element is found, the button is not added
         }
         var firstChild = code. firstChild;
         if (!firstChild) {
             return; // If the <code> element has no children, no button will be added
         }
         var button = document. createElement('button');
         button.textContent = '\uD83D\uDCCE'; // Use the 📎 symbol as the text of the "Copy" button
         button.style.position = 'relative';
         button.style.float = 'right';
         button.style.fontSize = '1em'; // optional: resize the button
         button.style.background = 'none'; // optional: remove the background color
         button.style.border = 'none'; // optional: remove the border
         button.style.cursor = 'pointer'; // optional: show pointer style
         button. addEventListener('click', function () {
             var range = document. createRange();
             range. selectNodeContents(code);
             range.setStartBefore(firstChild); // Set the range to before the first child node
             var selection = window. getSelection();
             selection. removeAllRanges();
             selection. addRange(range);

             try {
                 var success = document.execCommand('copy');
                 if (success) {
                     button.textContent = '\u2714';
                     setTimeout(function () {
                         button.textContent = '\uD83D\uDCCE'; // restore the button to "Copy"
                     }, 2000);
                 } else {
                     button.textContent = '\u2716';
                 }
             } catch (e) {
                 console. error(e);
                 button.textContent = '\u2716';
             }

             selection. removeAllRanges();
         });
         code.insertBefore(button, firstChild); // Insert the button before the first child element
     }

     function handleNewElements(mutationsList, observer) {
         for (var mutation of mutationsList) {
             if (mutation.type === 'childList') {
                 for (var node of mutation. addedNodes) {
                     if (node.nodeName === 'PRE') {
                         addCopyButton(node);
                     }
                 }
             }
         }
     }

     var observer = new MutationObserver(handleNewElements);
     observer. observe(document. documentElement, { childList: true, subtree: true });

     document.querySelectorAll('pre').forEach(addCopyButton);
})();