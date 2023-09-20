<script>
    import { navigate } from "svelte-routing";
    import { setListObject } from "./scripts";
    import Assets from './fields/assets.svelte';
    import Income from "./fields/income.svelte";
    import { darkMode } from "../stores";
    import ModeSwitch from "../ModeSwitch.svelte";
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

</script>

<h1 class:dark-mode={isDarkMode}>finGenie Create Profile</h1>
<div id="profile-menu">
    <button id="back-btn" on:click="{() => navigate("/")}">Back to Chat</button>
</div>

<div class="profile-form-container" class:dark-mode={isDarkMode} id="profile-form-container">
    <div class="profile-form">
        <form id="profile-form" class:dark-mode={isDarkMode}>
            <div class="profile-steps">
                <ul>
                    <li class="profile-step-menu1">
                        <span>1</span>
                        Basics
                    </li>
                    <li class="profile-step-menu2 active">
                        <span>2</span>
                        Assets
                    </li>
                    <li class="profile-step-menu3">
                        <span>3</span>
                        Debts
                    </li>
                    <li class="profile-step-menu4">
                        <span>4</span>
                        Confirm
                    </li>
                </ul>
            </div>

            <div class="profile-form-step-2 active" id="assets-step">
                <div class="profile-income-step" id="income">
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        Income
                    </div>
                    <Income/>
                </div>
                <div class="profile-asset-step" id="asset">
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        Assets
                    </div>
                    <Assets/>
                </div>
            </div>

            <div class="profile-form-btn-wrapper">
                <button class="profile-form-back-btn active" on:click="{() => navigate("/profile-form")}">
                    Back
                </button>
                <button class="profile-form-btn" on:click="{() => setListObject(1)}"
                                                 on:click="{() => setListObject(2)}"
                                                 on:click="{() => navigate("/profile-form/3")}">
                    Next Step
                </button>
            </div>

        </form>
    </div>
    <div style="position: absolute; right: -30px; width: 250px; height: 100%; top: 0;">
        <ModeSwitch />
    </div>
</div>
