<script>
    import SummaryTable from "./SummaryTable.svelte"
    import Summary from "./Summary.svelte"
    import { darkMode } from "../stores"
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

    export let value
    export let key
    export let userData
    export let item
</script>

{#if typeof (value) === 'object'}
    {#if value.length > 0 && key !== ''}
        <SummaryTable data={userData[key][item]}/>
    {:else if value.length > 0 && key == ''}
        <SummaryTable data={value}/>
    {:else}
        <Summary json={userData[key][item]} key={''} userData={userData}/>
    {/if}
{:else}
    <div class="profile-form-label" class:dark-mode={isDarkMode}>{value}</div>
{/if}
