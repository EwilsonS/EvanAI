<script>
    import { addField, setListType, setListAmount, setListObject } from "../scripts";
    import { darkMode } from "../../stores";

    let isDarkMode
    darkMode.subscribe((value) => {
        isDarkMode = value
    });
    let assetType;
    let assetAmount;
</script>

<div class="profile-input-text" class:dark-mode={isDarkMode}>
    <div>
        <input
            type="text"
            name="asset-type1"
            id="asset-type1"
            placeholder="Savings, Checkings, 401(k)"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={assetType}
            on:change="{() => setListType(2, assetType)}" />
    </div>
    <div>
        <input
            type="text"
            name="asset-amount1"
            id="asset-amount1"
            placeholder="$0.00"
            class="profile-input"
            class:dark-mode={isDarkMode}
            bind:value={assetAmount}
            on:change="{() => setListAmount(2, assetAmount)}"/>
    </div>
</div>
<div class="profile-input-text"
    class:dark-mode={isDarkMode}
    id="profile-asset-btn">
    <div>
        <button class="profile-table-add-btn"
            type="button" id="add-asset" on:click="{() => addField(2)}"
                                         on:click="{() => setListObject(2)}">+</button>
    </div>
</div>
