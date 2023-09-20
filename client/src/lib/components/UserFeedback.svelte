<script>
    import { copyMessage, setFeedback } from "../../chat_script2"

    export let feedbackActor
    export let componentIndex
    export let activity;
    let thumbUp
    let thumbDown
    export let feedbackUp = '';
    export let feedbackDown = '';

</script>
<svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0 user-scalable=no, user-scalable=0">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />
    <link rel="stylesheet" type="text/css" href="../styles.css">
</svelte:head>
<div id={`button-container-${componentIndex}`} class={`button-container ${activity}`}>
    <button id='copy-content' class={`material-symbols-outlined ${componentIndex}`} on:click={copyMessage}>content_copy</button>
    <button id={`${feedbackActor}-up-${componentIndex}`} on:click={event => setFeedback(event, componentIndex, true, thumbDown, 'thumb_down')}
            bind:this={thumbUp} class='material-symbols-outlined thumbUp-button'>thumb_up {`${feedbackUp}`}</button>
    <button id={`${feedbackActor}-down-${componentIndex}`} on:click={event => setFeedback(event, componentIndex, false, thumbUp, 'thumb_up')}
            bind:this={thumbDown} class='material-symbols-outlined thumbDown-button'>thumb_down {`${feedbackDown}`}</button>
</div>
