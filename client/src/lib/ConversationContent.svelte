<script>
import { onMount } from "svelte";
import { clearHistory } from "../chat_script";
import { conversationId } from "./stores";
import { navigate } from "svelte-routing";
import { darkMode } from "./stores";
let isDarkMode;

darkMode.subscribe((value) => {
    isDarkMode = value
});
let id;

let conversations = JSON.parse(localStorage.getItem("savedConversations"));
if (conversations) conversations.reverse();

conversationId.subscribe((value) => {
  id = value;
});

function continueConversation() {
    if (id <= -1) {
        return;
    }
    clearHistory();
    localStorage.setItem('user', JSON.stringify(conversations[id].user));
    localStorage.setItem('conversationHistory', JSON.stringify(conversations[id].history));
    localStorage.setItem('scores', JSON.stringify(conversations[id].scores));
}
</script>

<h1 style="margin-top: 0;"  class:dark-mode={isDarkMode}>finGenie</h1>
<div id="chat-container" class:dark-mode={isDarkMode}></div>
<form id="chat-form" class:dark-mode={isDarkMode}>
  <button on:click={(event) => {event.preventDefault(); continueConversation(); navigate('/');}}>Continue</button>
</form>
