<script>
    import { darkMode } from "../stores";
    import SummaryEntry from "./SummaryEntry.svelte";
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

    export let key
    export let json
    export let userData
    let mainKeys = Object.keys(json)
    if (key !== '') mainKeys = Object.keys(json[key]) //if a json object, get its keys

</script>

{#if mainKeys !== undefined}
    {#each mainKeys as item}
        <div class="profile-sum-label" class:dark-mode={isDarkMode}>{item.replace('_', ' ').toUpperCase()}</div>
        {#if key !== ''}
            <SummaryEntry value={json[key][item]} userData={userData} item={item} key={key}/>
        {:else}
            <SummaryEntry value={json[item]} userData={userData} item={item} key={key}/>
        {/if}
    {/each}
{/if}
