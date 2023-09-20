<svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>finGenie Profile</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="../../styles.css">
</svelte:head>

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
                    <li class="profile-step-menu2">
                        <span>2</span>
                        Assets
                    </li>
                    <li class="profile-step-menu3">
                        <span>3</span>
                        Debts
                    </li>
                    <li class="profile-step-menu4 active">
                        <span>4</span>
                        Confirm
                    </li>
                </ul>
            </div>

            <div class="profile-form-step-4 active" id="profile-step-menu4">
                <div class="profile-input-text" class:dark-mode={isDarkMode}>
                    Summary
                </div>
                <div id="form-summary"
                    style="font-size: 15px; font-weight: bold;"></div>
                <div class="profile-input-text" class:dark-mode={isDarkMode} id="summary-container">
                    <div id="formatted-summary" class="profile-form-label" class:dark-mode={isDarkMode} contenteditable="true">
                        <Summary key={"profile"} json={userData} userData={userData}/>
                    </div>
                </div>
            </div>

            <div class="profile-form-btn-wrapper">
                <button class="profile-form-back-btn active" on:click="{() => navigate("/profile-form/3")}">
                    Back
                </button>
                <button class="profile-form-btn" on:click="{() => localStorage.setItem("userSummary", "")}"
                                                 on:click="{() => navigate("/")}">
                    Submit
                </button>
            </div>

        </form>
    </div>
    <div style="position: absolute; right: -30px; width: 250px; height: 100%; top: 0;">
        <ModeSwitch />
    </div>
</div>

<script>
    import { navigate } from "svelte-routing";
    import { darkMode } from "../stores";
    import Summary from "./Summary.svelte"
    import ModeSwitch from "../ModeSwitch.svelte";
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

    let userData = JSON.parse(localStorage.getItem("userData"));

</script>
