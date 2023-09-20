<script>
    import { profiles, chosenProfile, showSearchResults } from "./stores";
    export let inputSearch

    let display = false
    let searchResults = []
    let savedProfiles = []

    showSearchResults.subscribe((value) => {
        display = value
    })

    const chooseProfile = (event) => {
        event.preventDefault()
        chosenProfile.set({id: event.target.id, name: event.target.innerHTML})
        showSearchResults.set(false)
    }

    profiles.subscribe(value => {
        let data = JSON.parse(value)
        if (data != null && data != undefined) {
            for (const [key, value] of Object.entries(data)) {
                let profile = value
                profile["id"] = key
                savedProfiles.push(profile)
            }
        }
    })

    const search = (query) => {
      let results = []
      let queryLowerCase = query.toLowerCase()
      savedProfiles.forEach(element => {
        if (element["first_name"].toLowerCase().includes(queryLowerCase) || element["last_name"].toLowerCase().includes(queryLowerCase) || (element["first_name"].toLowerCase() + " " + element["last_name"].toLowerCase()).includes(queryLowerCase)) {
          results.push(element)
        }
      });
      return results
    }

    $: {
        searchResults = search(inputSearch)
    }

</script>

<div id="search-results" class:hidden={!display}>
    {#each searchResults as result}
        <div style="padding: 1vh;"><button type="button" id={result["id"]} class="reset-button" on:click={(event) => chooseProfile(event)}>{result["first_name"]} {result["last_name"]}</button></div>
    {/each}
</div>
