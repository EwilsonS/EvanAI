<script>
  import { clearHistory } from "../../chat_script2";
  let selectedLanguage = localStorage.getItem("display_language") || "en";

  function handleLanguageChange(event) {
    selectedLanguage = event.target.value;
    localStorage.setItem("display_language", selectedLanguage);
    clearHistory()
    location.reload()
  }
</script>

<div id="language-div">
  <label>
    <input
      type="radio"
      name="language"
      value="en"
      checked
      bind:group={selectedLanguage}
      on:change={handleLanguageChange}
    />
    English
  </label>
  <label>
    <input
      type="radio"
      name="language"
      value="fr"
      bind:group={selectedLanguage}
      on:change={handleLanguageChange}
    />
    Français
  </label>
  <label>
    <input
      type="radio"
      name="language"
      value="es"
      bind:group={selectedLanguage}
      on:change={handleLanguageChange}
    />
    Español
  </label>
</div>

<style>
  #language-div {
    color: rgb(230, 230, 230);
    font-size: smaller;
    position: fixed;
    right: 20%;

  }

  @media (max-width: 475px) {
    #language-div {
      font-size: xx-small;
      position: fixed;
      right: 20%;

  }

  }

</style>
