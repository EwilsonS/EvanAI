<script>
    import { onMount } from "svelte";
    import { navigate } from "svelte-routing";
    import { loadProfiles, showProfileSummary, handleClick, handleMouseOut, clearUserProfile, generateUserProfileSummary } from "../profile_script.js";

    onMount(() => {
        loadProfiles();
        // Add the event listener to the window object
        window.addEventListener('storage', handleStorageEvent);

        function handleStorageEvent(event) {
            console.log('Storage event detected:', event);

            // Check if userProfile has been changed/submitted
            if (event.key === 'userData') {
                // Update the UI theme based on the new user profile
                localStorage.setItem("userSummary", "Loading...");
                generateUserProfileSummary();
                console.log("NEW USER SUMMARY", localStorage.getItem("userSummary"));
                document.getElementById("user-profile").classList.remove('inactive');
                document.getElementById("clear-user-history").classList.remove('inactive');
            }
        }
        if(localStorage.getItem("userData")){
            document.getElementById("user-profile").classList.remove('inactive');
            document.getElementById("clear-user-history").classList.remove('inactive');
        }
    });
</script>

<div id="profile-menu">
    <button id="create-profile" on:click="{() => navigate('/profile-form')}">Create Profile</button>
    <button id="restore-conversation" on:click="{() => navigate('/restore-conversation')}">Restore Conversation</button>

    <div id="profiles">
        <h2 id="sample-profile-text">Sample Profiles</h2>
        <button class="profile" id="profile-0" on:mouseenter={() => showProfileSummary("profile-0")}
                                               on:click={(event) => handleClick(event)}
                                               on:mouseleave={() => handleMouseOut}>Juan</button>
        <button class="profile" id="profile-1" on:mouseenter={() => showProfileSummary("profile-1")}
                                               on:click={(event) => handleClick(event)}
                                               on:mouseleave={() => handleMouseOut}>Jane</button>
        <button class="profile" id="profile-2" on:mouseenter={() => showProfileSummary("profile-2")}
                                               on:click={(event) => handleClick(event)}
                                               on:mouseleave={() => handleMouseOut}>John</button>
        <button class="profile" id="profile-3" on:mouseenter={() => showProfileSummary("profile-3")}
                                               on:click={(event) => handleClick(event)}
                                               on:mouseleave={() => handleMouseOut}>Emily</button>
        <br>
        <button class="profile inactive" id="user-profile" on:mouseenter={() => showProfileSummary()}
                                                           on:click={(event) => handleClick(event)}
                                                           on:mouseleave={() => handleMouseOut}>My Profile</button>
        <br>
    </div>
    <br>
    <button class="main-btn inactive" type="button" id="clear-user-history" on:click={() => clearUserProfile()}>Clear My Profile</button>

    <div id="summary"></div>
</div>
