<script>
    import { darkMode } from "../stores";
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });
    export let keys
    export let value
</script>

{#each keys as key}
    {#if key !== ''}
        <td class:dark-mode={isDarkMode}>{value[key]}</td>
    {/if}
{/each}
