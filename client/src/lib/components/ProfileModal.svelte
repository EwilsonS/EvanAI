<script>
  export let showProfileModal; // boolean
  import { writable } from "svelte/store";
  let loading = writable(true);

  let dialog; // HTMLDialogElement

  $: if (dialog && showProfileModal) {
    dialog.showModal();
    //document.getElementById('summary-p').addEventListener('innerHTMLChange', updateLoader)
  }

  function updateLoader() {
    loading.update((n) => !n);
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
<dialog
  bind:this={dialog}
  on:close={() => (showProfileModal = false)}
  on:click|self={() => (document.getElementById("summary-p").innerHTML = "")}
  on:click|self={() => dialog.close()}
>
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div on:click|stopPropagation>
    <div id="popup-header-container">
      <h3>
        <small>Profile</small>
      </h3>
      <button
        id="close-btn"
        class="material-symbols-outlined"
        on:click={() => dialog.close()}>close</button
      >
    </div>

    <hr style="border-width: 0.5px" />
    <div id="popup-email-div">
      <h3>
        <small style="color:#d9d9e3">Email:</small>
      </h3>
      <input type="text" id="email" name="email" placeholder="Email" />
    </div>
    <div id="popup-password-container">
      <h3><small style="color:#d9d9e3">Password:</small></h3>
      <input type="text" id="password" name="password" placeholder="Password" />
    </div>
    <!-- svelte-ignore a11y-autofocus -->
  </div>
</dialog>
