<script>
    import { addField, setListAmount, setListObject, setListType } from "../scripts";
    import { darkMode } from "../../stores";

    let isDarkMode
    darkMode.subscribe((value) => {
        isDarkMode = value
    });

    let dependentType;
    let dependentAge;

</script>

<div class="profile-input-text" class:dark-mode={isDarkMode} id="dependents">
    <div>
        <input
            type="text"
            name="dependent-type"
            id="dependent-type"
            placeholder="Son, Daughter, Step, etc."
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={dependentType}
            on:change="{() => setListType(4, dependentType)}"/>
    </div>
    <div>
        <input
            type="text"
            name="dependent-amount"
            id="dependent-amount"
            placeholder="0"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={dependentAge}
            on:change="{() => setListAmount(4, dependentAge)}"/>
    </div>
</div>
<div class="profile-input-text"
    class:dark-mode={isDarkMode}
    id="profile-dependent-btn">
    <div>
        <button class="profile-table-add-btn"
            type="button" id="add-dependent" on:click="{() => addField(4)}"
                                             on:click="{() => setListObject(4)}">+</button>
    </div>
</div>
