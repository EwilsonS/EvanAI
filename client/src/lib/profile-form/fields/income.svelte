<script>
    import { addField, setListType, setListAmount, setListObject } from "../scripts";
    import { darkMode } from "../../stores";

    let isDarkMode
    darkMode.subscribe((value) => {
        isDarkMode = value
    });
    let incomeType;
    let incomeAmount;
</script>

<div class="profile-input-text" class:dark-mode={isDarkMode}>
    <div>
        <input
            type="text"
            name="income-type1"
            id="income-type1"
            placeholder="Salary, Side hussle, Pension"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={incomeType}
            on:change="{() => setListType(1, incomeType)}" />
    </div>
    <div>
        <input
            type="text"
            name="income-amount1"
            id="income-amount1"
            placeholder="$0.00"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={incomeAmount}
            on:change="{() => setListAmount(1, incomeAmount)}" />
    </div>
</div>
<div class="profile-input-text"
    class:dark-mode={isDarkMode}
    id="profile-income-btn">
    <div>
        <button class="profile-table-add-btn"
            type="button" id="add-income" on:click="{() => addField(1)}"
                                          on:click="{() => setListObject(1)}">+</button>
    </div>
</div>
