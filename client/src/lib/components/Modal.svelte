<script>
	import { copyConversation, copySummary } from "../../chat_script2";
	export let showModal; // boolean
	import { writable  }from 'svelte/store';
	import { langOpts } from "../../switch_language";
	let loading = writable(true)

	let dialog; // HTMLDialogElement

	$: if (dialog && showModal) {
		dialog.showModal()
		//document.getElementById('summary-p').addEventListener('innerHTMLChange', updateLoader)
	}

	function updateLoader () {
		loading.update(n => !n)
	}

	const lang = localStorage.getItem("display_language") || 'en'
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
<dialog
	bind:this={dialog}
	on:close={() => (showModal = false)}
	on:click|self={() => document.getElementById('summary-p').innerHTML = ''}
	on:click|self={() => dialog.close()}
>
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div on:click|stopPropagation>
		<div id="popup-header-container">
			<h3>
				<small>{langOpts.chatOptions[lang]}</small>
			</h3>
			<button id="close-btn" class="material-symbols-outlined" on:click={() => dialog.close()}>close</button>
		</div>

		<hr style="border-width: 0.5px"/>
		<div id="popup-copy-div">
			<h3>
				<small style="color:white; display:flex">{langOpts.copyConversation[lang]}</small>
			</h3>
			<div style="color:#d9d9e3;">
				<button style="background-color:#444654; margin-left:40px;"
				class="material-symbols-outlined" id="copy-content"
				on:click={copyConversation}>content_copy</button>
			</div>
		</div>
		<div id="popup-summary-container">
			<div style="display:flex; justify-content:space-between; margin-right:10px;">
				<h3><small style="color:#d9d9e3">{langOpts.chatSummary[lang]}</small></h3>
				<div style="color:#d9d9e3;">
					<button style="background-color:#33343f; margin-left:40px;"
					class="material-symbols-outlined" id="copy-content"
					on:click={copySummary}
					>content_copy</button>
				</div>
			</div>
			<div id="popup-summary-text" class="bot-message">
			</div>
        </div>
		<!-- svelte-ignore a11y-autofocus -->
	</div>
</dialog>
