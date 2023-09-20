<svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0 user-scalable=no, user-scalable=0">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />
    <link rel="stylesheet" type="text/css" href="../styles.css">
</svelte:head>
<script>
    import { onMount } from "svelte";
    export let sender
    let icon
    let userIcon

    $: if (sender === 'bot') {
        icon = 'smart_toy'
    } else {
        icon = 'person'
    }

</script>
<div id={`${sender}-icon`} class='material-symbols-outlined'>
    {`${icon}`}
</div>
