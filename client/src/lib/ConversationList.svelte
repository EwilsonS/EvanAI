<script>
    import { onMount } from "svelte";
    import { addMessage, clearTimeouts, displayScore } from "../chat_script";
    import { navigate } from "svelte-routing";
    import { conversationId } from "./stores";

    let conversations = JSON.parse(localStorage.getItem("savedConversations"));
    if (conversations) conversations.reverse();

    onMount(() => {clearTimeouts();})

    function handleClick(event, index, c) {
        clearTimeouts();
        let buttons = document.querySelectorAll(".conversation");
        buttons.forEach(function(b) {
            if (b.id != index.toString()) b.classList.remove("clicked");
        });
        conversationId.set(index);
        event.target.classList.toggle("clicked");
        document.getElementById("chat-container").innerHTML = '';
        displayScore(c.scores);
        c.history.forEach(message => {
            addMessage(message.content, message.role === "user" ? "user" : "bot");
        });
    }
</script>

<svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>finGenie</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="../styles.css">
</svelte:head>

<div id="profile-menu">
    <h2>Saved Conversations</h2>
    {#if conversations}
        {#each conversations as c, index}
            {#if c.history.length > 0}
                {#if c.user == null}
                    <button id={index.toString()} class="conversation" on:click={(event) => handleClick(event, index, c)}>Unknown</button>
                {:else}
                    <button id={index.toString()} class="conversation" on:click={(event) => handleClick(event, index, c)}>{c.user.name}</button>
                {/if}
            {/if}
        {/each}
    {:else}
        <p style="width: 70%; margin: auto; text-align: center; color: #9facba">There are no saved conversations</p>
    {/if}
    <button id="back-btn" on:click="{() => navigate('/')}">Back to Chat</button>
</div>
