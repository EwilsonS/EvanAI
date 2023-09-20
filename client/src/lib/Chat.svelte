<script>
  import { onMount } from "svelte";
  import { sendMessage, loadConversationHistory, clearHistory, clearSuggestions } from "../chat_script.js";
  import { darkMode } from "./stores";
  let isDarkMode;

  darkMode.subscribe((value) => {
      isDarkMode = value
  });
  // Load the conversation history when the page loads
  onMount(() => {
      loadConversationHistory();
    });
</script>

<svelte:head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>finGenie</title>
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" type="text/css" href="../styles.css">
</svelte:head>
<h1 style="margin-top: 0;" class:dark-mode={isDarkMode}>finGenie</h1>
<div id="chat-container" class:dark-mode={isDarkMode}></div>
<form id="chat-form" class:dark-mode={isDarkMode}>
  <input type="text" id="message" name="message" class:dark-mode={isDarkMode} required placeholder="Type your message..." on:input={() => clearSuggestions()}>
  <button type="submit" on:click={(event) => sendMessage(event)}>Send</button>
  <button type="button" id="clear-history" on:click={() => clearHistory(localStorage.getItem("user"))}>Clear History</button>
</form>
