<script>
    import { onMount } from "svelte";
    import { handleEnter, loadConversationHistory, sendMessage,
             clearHistory, clearSuggestions, generateConversationSummary} from "../chat_script2"
    import Modal from './components/Modal.svelte';
    import SearchResults from "./SearchResults.svelte";
    import { baseUrl, creds } from "../chat_script2";
    import { chosenProfile, profiles, showSearchResults } from "./stores";
    import { langOpts } from "../switch_language";

	  let showModal = false;
    let profile = "Choose a profile"
    let summary = null

    chosenProfile.subscribe((value) => {
        let input = document.getElementById("profile-search")
        if (input instanceof HTMLInputElement) {
          if (Object.keys(value).includes("name")) {
              input.value = value["name"]
              summary = JSON.parse(localStorage.getItem("profiles"))[value["id"]]["summary"]
          }
          else
              input.value = "Choose a profile"
        }
    })

    const box = document.getElementById('text-field-container')
    let inputSearch = ""

    const handleFocus = () => {
      if (navigator.userAgent.match(/webOS/i)
         || navigator.userAgent.match(/iPhone/i)
         || navigator.userAgent.match(/iPad/i)) {
          document.getElementById('chat-box').style.visibility = 'hidden'
      }
    }

    function loadProfiles (_callback) {
      fetch(baseUrl + '/profiles', {
        method: 'GET',
        headers: {
          Authorization: 'Basic ' + creds,
          Accept: 'application/json'
        }
      })
      .then(response => response.json())
      .then(data => {
        localStorage.setItem("profiles", JSON.stringify(data))
        profiles.set(JSON.stringify(data))
      })
    }

    const handleBlur = () => {
      document.getElementById('chat-box').style.visibility = 'visible'
    }

    // Load the conversation history when the page loads
    onMount(() => {
      loadConversationHistory()
      if (localStorage.getItem("profiles") == null)
        loadProfiles()
      else
        profiles.set(localStorage.getItem("profiles"))
    });

    $: {
      summary
    }

    const queryParams = new URLSearchParams(location.search);
    queryParams.set('agent', 'agent');
    history.replaceState({}, '', `${location.pathname}?${queryParams}`);

    const lang = localStorage.getItem("display_language") || "en";
  </script>

  <svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0 user-scalable=no;user-scalable=0;">
    <title></title>
    <link href="https://fonts.googleapis.com/css?family=Raleway" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />
    <link rel="stylesheet" type="text/css" href="../styles.css">
  </svelte:head>
  <div id="header-container">
    <h4>finGenie</h4>
    <div>
      <div style="position: absolute; left: 3vw; top: 1vw;">
        <input type="text" id="profile-search" name="profile-search" placeholder={profile} on:input={() => showSearchResults.set(true)} bind:value={inputSearch}/>
      </div>
      <SearchResults inputSearch={inputSearch}/>
    <div style="margin-left:45vw"><button id="options-btn" class="material-symbols-outlined"
      on:click={() => (showModal = true)}
      on:click={generateConversationSummary}
      > more_vert </button></div>
    </div>
    <Modal bind:showModal>


    </Modal>
  </div>
  <div id="chat-box">
    <div id="chat-container2">
      {#if summary != null}
        <div id="summary-box">{summary}</div>
      {/if}
    </div>
  </div>
  <div id="text-field-container">
    <div id="hide-bar">
    </div>
    <form on:submit|preventDefault={handleEnter} id="text-field-box">
        <input type="text" id="message" name="message" placeholder={langOpts.inputPlaceholder[lang]} on:input={clearSuggestions} on:focus={handleFocus} on:blur={handleBlur}>
        <button type="submit" id="submit-button" on:click={sendMessage} on:click={handleBlur}><i class="material-icons">send</i></button>
        <button type="button" id="clear-history" on:click={clearHistory} on:click={handleBlur}><i class="material-symbols-outlined">delete</i></button>
    </form>
  </div>
