<svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0 user-scalable=no, user-scalable=0">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />
    <link rel="stylesheet" type="text/css" href="../styles.css">
</svelte:head>

<script>
    import { handleEnter, sendMessage, clearSuggestions,
             clearHistory } from "../../chat_script2";
    import { langOpts } from "../../switch_language";

    const handleFocus = () => {
        if (navigator.userAgent.match(/webOS/i)
            || navigator.userAgent.match(/iPhone/i)
            || navigator.userAgent.match(/iPad/i)) {
            document.getElementById('chat-box').style.visibility = 'hidden'
        }
        }

    const handleBlur = () => {
      document.getElementById('chat-box').style.visibility = 'visible'
    }

    const lang = localStorage.getItem("display_language") || "en";

    const clearAndReload = () => {
        clearHistory()
        location.reload()
    }
</script>

<form on:submit|preventDefault={handleEnter} id="text-field-box">
    <input type="text" id="message" name="message" placeholder={langOpts.inputPlaceholder[lang]} on:input={clearSuggestions} on:focus={handleFocus} on:blur={handleBlur}>
    <button type="submit" id="submit-button" on:click={sendMessage} on:click={handleBlur}><i class="material-icons">send</i></button>
    <button type="button" id="clear-history" on:click={clearAndReload} on:click={handleBlur}><i class="material-symbols-outlined">delete</i></button>
</form>
