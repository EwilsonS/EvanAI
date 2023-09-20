<script>
  import { onMount } from "svelte";
  import {
    loadConversationHistory,
    generateConversationSummary,
    trackAnalyticsEvent,
    eventHamburgerMenu,
  } from "../chat_script2";
  import Modal from "./components/Modal.svelte";
  import ProfileModal from "./components/ProfileModal.svelte";
  import UserInput from "./components/UserInput.svelte";
  import SelectAgent from "./components/SelectAgent.svelte";
  import ApiCall from "./components/ApiCall.svelte";
  import SelectLanguage from "./components/SelectLanguage.svelte";

  export let agent = ''

  let showModal = false;
  let showList =false;
  let showProfileModal = false;
  const box = document.getElementById("text-field-container");

  // Load the conversation history when the page loads
  onMount(() => loadConversationHistory());

  const generateEventHamburgerMenu = () => {
    trackAnalyticsEvent(eventHamburgerMenu);
  };

  const queryParams = new URLSearchParams(location.search);
  queryParams.set('agent', agent);
  history.replaceState({}, '', `${location.pathname}?${queryParams}`);
</script>

<svelte:head>
  <meta charset="UTF-8" />
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1.0 user-scalable=no, user-scalable=0"
  />
  <title />
  <link
    href="https://fonts.googleapis.com/css?family=Raleway"
    rel="stylesheet"
  />
  <link
    href="https://fonts.googleapis.com/icon?family=Material+Icons"
    rel="stylesheet"
  />
  <link
    rel="stylesheet"
    href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0"
  />
  <link rel="stylesheet" type="text/css" href="../styles.css" />
</svelte:head>
<div id="header-container">
  <h4>finGenie</h4>
  <div style="margin-left:45vw">
    <SelectLanguage />
    <button
      id="options-btn"
      class="material-symbols-outlined"
      on:click={() => (showModal = true)}
      on:click={generateEventHamburgerMenu}
      on:click={generateConversationSummary}
    >
      more_vert
    </button>
    <button
      id="profile-btn"
      class="material-symbols-outlined"
      on:click={() => (showProfileModal = true)}
    >
      account_circle
    </button>
  </div>

  <Modal bind:showModal />
  <ApiCall bind:showList></ApiCall>

</div>
<div id="chat-box">
  <div id="chat-container2" />
</div>
<div id="text-field-container">
  <div id="hide-bar" />
  <UserInput />
</div>
