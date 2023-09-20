<script>
    import { addField, setListType, setListAmount, setLoanPrincipal, setListObject } from "../scripts";
    import { darkMode } from "../../stores";

    let isDarkMode
    darkMode.subscribe((value) => {
        isDarkMode = value
    });
    let loanType;
    let loanAmount;
    let loanPrincipal;
</script>

<div class="profile-input-text" class:dark-mode={isDarkMode}>
    <div>
        <input
            type="text"
            name="loan-type1"
            id="loan-type1"
            placeholder="Home Mortgage, car, Student Loans"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={loanType}
            on:change="{() => setListType(3, loanType)}" />
    </div>
    <div>
        <input
            type="text"
            name="loan-amount1"
            id="loan-amount1"
            placeholder="$0.00"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={loanAmount}
            on:change="{() => setListAmount(3, loanAmount)}"/>
    </div>
    <div>
        <input
            type="text"
            name="loan-principal1"
            id="loan-principal1"
            placeholder="$0.00"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={loanPrincipal}
            on:change="{() => setLoanPrincipal(loanPrincipal)}"/>
    </div>
</div>
<div class="profile-input-text"
    class:dark-mode={isDarkMode}
    id="profile-loan-btn">
    <div>
        <button class="profile-table-add-btn"
            type="button" id="add-loan" on:click="{() => addField(3)}"
                                        on:click="{() => setListObject(3)}">+</button>
    </div>
</div>
