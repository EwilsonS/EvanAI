<script>
  import { getCardData, getSystemCards, saveNewData } from "../../chat_script2";
  export let showList; // boolean
  //   import { writable } from "svelte/store";
  //   let loading = writable(true);
  let arr = [];
  let dialog; // HTMLDialogElement
  let res = null;
  let fileText = null;
  let fileName = "";

  async function fetchData() {
    res = await getSystemCards();
    arr = Object.values(res);
  }

  let activeButton = -1
  async function textData(args, index) {
	activeButton = index;
    fileName = args;
    fileText = await getCardData(args);
  }

  const handleInput = (event) => {
    fileText = event.currentTarget.value;
  };

  const saveNewFile = (fileName, fileText) => {
    saveNewData(fileName, fileText);
    dialog.close();
  };

  $: if (dialog && showList) {
    fetchData();
    dialog.showModal();
  }
</script>
<style>
	 .active {
    background-color: #0b0b0b;
    color: #fff;
  }
</style>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
<dialog
  bind:this={dialog}
  on:close={() => (showList = false , activeButton =-1 , fileText ='')}
  on:click|self={() => (document.getElementById("summary-p").innerHTML = "")}
  on:click|self={() => dialog.close()}
>
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div on:click|stopPropagation>
    <div id="popup-header-container">
      <h3>
        <small>System Cards</small>
      </h3>
      <button
        id="close-btn"
        class="material-symbols-outlined"
        on:click={() => dialog.close()}>Close</button
      >
    </div>

    <hr style="border-width: 0.5px" />
    <ul>
      {#each arr as item, index}
        <button class={activeButton === index ? 'active' : ''} on:click={() => textData(arr[index], index)}>{arr[index]}</button>
      {/each}
    </ul>
    <div>
      <textarea rows="11" cols="60" on:input={handleInput}>{fileText}</textarea>
    </div>

    <button on:click={() => saveNewFile(fileName, fileText)}>Save</button>
    <!-- svelte-ignore a11y-autofocus -->
  </div>
</dialog>
