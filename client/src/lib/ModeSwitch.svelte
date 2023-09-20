<script>
    import { onMount } from "svelte";
    import { darkMode } from "./stores.js"

    onMount(() => applyMode());

    let isDarkMode;
    darkMode.subscribe((value) => {
        isDarkMode = value
    })

    function applyMode() {
        if (!isDarkMode) {
            document.getElementsByTagName('body')[0].style.backgroundColor = '#fff'
            document.getElementById('mode-switch').innerHTML = '&#9728;&#65039;'
            const botMessages = document.querySelectorAll('.bot-message .message-text')
            botMessages.forEach(function (message) {
                message.classList.toggle('dark-mode')
            })
        } else {
            document.getElementsByTagName('body')[0].style.backgroundColor = '#1C2127'
            document.getElementById('mode-switch').innerHTML = '&#127761;'
            const botMessages = document.querySelectorAll('.bot-message .message-text')
            botMessages.forEach(function (message) {
                message.classList.toggle('dark-mode')
            })
        }
    }

    function switchMode () {
        darkMode.update((value) => !value)
        applyMode()
    }
</script>

<button type="button" id="mode-switch" value="light" on:click={switchMode} class:dark-mode={isDarkMode}>&#9728;&#65039;</button>
