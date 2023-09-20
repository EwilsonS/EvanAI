<script>
    import { navigate } from "svelte-routing";
    import Loans from "./fields/loans.svelte";
    import { addExpenses, setListObject } from "./scripts";
    import { darkMode } from "../stores";
    import ModeSwitch from "../ModeSwitch.svelte";
    let isDarkMode;

    darkMode.subscribe((value) => {
        isDarkMode = value
    });

    let creditCard;
    let rent;
    let gas;
    let phone;
    let utilities;
    let entertainment;
    let food;
    let childcare;
    let other;

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
                    <li class="profile-step-menu2">
                        <span>2</span>
                        Assets
                    </li>
                    <li class="profile-step-menu3 active">
                        <span>3</span>
                        Debts
                    </li>
                    <li class="profile-step-menu4">
                        <span>4</span>
                        Confirm
                    </li>
                </ul>
            </div>

            <div class="profile-form-step-3 active" id="debts-step">
                <div class="profile-income-step">
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        <div>
                            <label for="credit-card"
                                class="profile-form-label" class:dark-mode={isDarkMode}> Credit Card
                                Debt: </label>
                            <input
                                type="text"
                                name="credit-card"
                                placeholder="$0.00"
                                id="credit-card"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={creditCard}/>
                        </div>
                    </div>
                </div>
                <div class="profile-loan-step" id="loan">
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        Loans
                    </div>
                    <Loans/>
                </div>
                <div class="profile-income-step">
                    Monthly Expenses
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        <div>
                            <label for="rent" class="profile-form-label" class:dark-mode={isDarkMode}>
                                Rent </label>
                            <input
                                type="text"
                                name="rent"
                                placeholder="$0.00"
                                id="rent"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={rent}/>
                        </div>
                    </div>
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        <div>
                            <label for="gas"
                                class="profile-form-label" class:dark-mode={isDarkMode}> Gas
                            </label>
                            <input
                                type="text"
                                name="gas"
                                placeholder="$0.00"
                                id="gas"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={gas} />
                        </div>
                        <div>
                            <label for="phone"
                                class="profile-form-label" class:dark-mode={isDarkMode}> Phone
                            </label>
                            <input
                                type="text"
                                name="phone"
                                placeholder="$0.00"
                                id="phone"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={phone} />
                        </div>
                        <div>
                            <label for="utilities"
                                class="profile-form-label" class:dark-mode={isDarkMode}>
                                Utilities </label>
                            <input
                                type="text"
                                name="utilities"
                                placeholder="$0.00"
                                id="utilities"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={utilities} />
                        </div>
                    </div>
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        <div>
                            <label for="entertainment"
                                class="profile-form-label" class:dark-mode={isDarkMode}>
                                Entertainment </label>
                            <input
                                type="text"
                                name="entertainment"
                                placeholder="$0.00"
                                id="entertainment"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={entertainment} />
                        </div>
                        <div>
                            <label for="food" class="profile-form-label" class:dark-mode={isDarkMode}>
                                Food </label>
                            <input
                                type="text"
                                name="food"
                                placeholder="$0.00"
                                id="food"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={food} />
                        </div>
                        <div>
                            <label for="childcare"
                                class="profile-form-label" class:dark-mode={isDarkMode}> Childcare
                            </label>
                            <input
                                type="text"
                                name="childcare"
                                placeholder="$0.00"
                                id="childcare"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={childcare} />
                        </div>
                    </div>
                    <div class="profile-input-text" class:dark-mode={isDarkMode}>
                        <div>
                            <label for="other"
                                class="profile-form-label" class:dark-mode={isDarkMode}> Other
                            </label>
                            <input
                                type="text"
                                name="other"
                                placeholder="$0.00"
                                id="other"
                                class="profile-input"
                                class:dark-mode={isDarkMode}
                                bind:value={other} />
                        </div>
                    </div>
                </div>
            </div>

            <div class="profile-form-btn-wrapper">
                <button class="profile-form-back-btn active" on:click="{() => navigate("/profile-form/2")}">
                    Back
                </button>
                <button class="profile-form-btn" on:click="{() => setListObject(3)}"
                                                 on:click="{() => addExpenses(creditCard, rent, gas, phone, utilities,
                                                                              entertainment, food, childcare, other)}"
                                                 on:click="{() => navigate("/profile-form/4")}">
                    Next Step
                </button>
            </div>

        </form>
    </div>
    <div style="position: absolute; right: -30px; width: 250px; height: 100%; top: 0;">
        <ModeSwitch />
    </div>
</div>
