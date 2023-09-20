<script>
    import { darkMode } from "../stores";
    import SummaryData from "./SummaryData.svelte";
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

    export let data
    let length = data[0].length
    let headerList = Object.keys(data[0])

</script>

{#if length !== 0}
<table class='summary-table'>
    <tr class:dark-mode={isDarkMode}></tr>
    {#each headerList as header}
        <th class:dark-mode={isDarkMode}>{header.replace('_', ' ')}</th>
    {/each}
    {#each data as value}
        <tr class:dark-mode={isDarkMode}></tr>
        <SummaryData keys={Object.keys(value)} value={value} />
    {/each}
</table>
{/if}
