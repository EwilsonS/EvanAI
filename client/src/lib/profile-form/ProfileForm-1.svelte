<script>
    import { Router, Link, Route, navigate } from "svelte-routing";
    import { addBasics, setListObject, addField } from "./scripts";
    import ModeSwitch from "../ModeSwitch.svelte";
    import Dependents from "./fields/dependents.svelte";
    import { onMount } from "svelte";
    import { darkMode } from "../stores";
    //NOTE bind:value={valueName}

    let age = "";
    let firstname = "";
    let lastname = "";
    let maritalstatus = "";
    let employment = "";
    let dependents = [];
    let dependent = JSON.parse('{ "type": "", "age": ""}');
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

</script>

<div id="profile-menu">
    <button id="back-btn" on:click="{() => navigate("/")}">Back to Chat</button>
</div>
<h1 class:dark-mode={isDarkMode}>finGenie Create Profile</h1>
<div class="profile-form-container" class:dark-mode={isDarkMode} id="profile-form-container">
    <div class="profile-form">
        <form id="profile-form" class:dark-mode={isDarkMode}>
            <div class="profile-steps">
                <ul>
                    <li class="profile-step-menu1 active">
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
                    <li class="profile-step-menu4">
                        <span>4</span>
                        Confirm
                    </li>
                </ul>
            </div>
            <div class="profile-form-step-1 active" id="basics-step">
                <div class="profile-input-text" class:dark-mode={isDarkMode}>
                    <div>
                        <label for="firstname"
                            class="profile-form-label" class:dark-mode={isDarkMode} id="firstname-label" > First name
                        </label>
                        <input
                            type="text"
                            name="firstname"
                            placeholder="John"
                            id="firstname"
                            class="profile-input"
                            class:dark-mode={isDarkMode}
                            bind:value={firstname}/>
                    </div>
                    <div>
                        <label for="lastname" class="profile-form-label" class:dark-mode={isDarkMode}>
                            Last name </label>
                        <input
                            type="text"
                            name="lastname"
                            placeholder="Doe"
                            id="lastname"
                            class="profile-input"
                            class:dark-mode={isDarkMode}
                            bind:value={lastname}/>
                    </div>
                </div>
                <div class="profile-input-text" class:dark-mode={isDarkMode}>
                    <div>
                        <label for="dob" class="profile-form-label" class:dark-mode={isDarkMode}>
                            Date of Birth </label>
                        <input
                            type="date"
                            name="dob"
                            id="dob"
                            class="profile-input"
                            class:dark-mode={isDarkMode}
                            bind:value={age}/>
                    </div>
                </div>
                <div class="profile-input-text" class:dark-mode={isDarkMode}>
                    <div>
                        <label for="maritalstatus"
                            class="profile-form-label" class:dark-mode={isDarkMode}> Marital Status
                        </label>
                        <select
                            bind:value={maritalstatus}
                            name="maritalstatus"
                            placeholder="Married"
                            id="maritalstatus"
                            class="profile-input-select"
                            class:dark-mode={isDarkMode}>
                            <option value="single">Single</option>
                            <option value="married">Married</option>
                        </select>
                    </div>
                    <div>
                        <label for="employment"
                            class="profile-form-label" class:dark-mode={isDarkMode}> Employment
                        </label>
                        <select
                            bind:value={employment}
                            name="employment"
                            id="employment"
                            class="profile-input-select"
                            class:dark-mode={isDarkMode}>
                            <option value="employed">Employed</option>
                            <option value="self-employed">Self-employed</option>
                            <option value="unemployed">Unemployed</option>
                        </select>
                    </div>
                </div>
                <div class="profile-income-step" id="dependent">
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        Dependents
                    </div>
                    <Dependents/>
                </div>
            </div>

            <div class="profile-form-btn-wrapper">
                <button class="profile-form-back-btn">
                    Back
                </button>
                <button class="profile-form-btn"
                on:click="{() => addBasics(age, firstname, lastname, maritalstatus, employment)}"
                on:click="{() => setListObject(4)}"
                on:click="{() => navigate("/profile-form/2")}" >
                    Next Step
                </button>
            </div>
        </form>
    </div>
    <div style="position: absolute; right: -30px; width: 250px; height: 100%; top: 0;">
        <ModeSwitch />
    </div>
</div>
