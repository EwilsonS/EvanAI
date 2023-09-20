<!--
  This component is a button to help toggle between views/agents.
  Add the component below with boolean prop to whichever components
  you want to toggle between. Add vars and <q> tags to this as necessary

  <SelectAgent isAdvisor={true} />
-->

<script>
  import { clearHistory } from "../../chat_script2"

  export let isEducate = false
  export let isAdvisor = false
  export let isOldChat = false

</script>

<a href='/' on:click={clearHistory} class='select-agent-btn {isEducate ? 'active-btn': ''}'>educate</a>
<a href='/advisor' on:click={clearHistory} class='select-agent-btn {isAdvisor ? 'active-btn': ''}'>advisor</a>
<a href='/prompt' on:click={clearHistory} class='select-agent-btn {isEducate ? 'active-btn': ''}'>educate</a>
<a href='/chat' on:click={clearHistory} class='select-agent-btn {isOldChat ? 'active-btn': ''}'>old UI</a>

<style>
  .select-agent-btn {
    color: white;
    background-color: black;
    margin-left: 5px;
    font-size: small;
    text-decoration: none;
    padding: 5px;
    transition: .2s;
  }

  .select-agent-btn:hover  {
    background-color: rgb(83, 83, 83);
  }

  .active-btn  {
    background-color: rgb(164, 2, 2);
  }
</style>
