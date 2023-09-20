<script>
  import Chat from './lib/Chat.svelte'
  import Chat2 from './lib/Chat2.svelte';
  import AdvisorView from './lib/AdvisorView.svelte';
  import ProfileMenu from "./lib/ProfileMenu.svelte";
  import Score from "./lib/Score.svelte"
  import ProfileForm1 from './lib/profile-form/ProfileForm-1.svelte';
  import ProfileForm2 from './lib/profile-form/ProfileForm-2.svelte';
  import ProfileForm3 from './lib/profile-form/ProfileForm-3.svelte';
  import ProfileForm4 from './lib/profile-form/ProfileForm-4.svelte';
  import ModeSwitch from "./lib/ModeSwitch.svelte";
  import { Router, Link, Route } from "svelte-routing";
  import ConversationList from './lib/ConversationList.svelte';
  import ConversationContent from './lib/ConversationContent.svelte';
  import { onMount } from 'svelte'
  import { navigate } from 'svelte-routing'
  import { logEvent } from 'firebase/analytics'
  import analytics from './firebase.js'
    import PromptView from './lib/PromptView.svelte';

  onMount(() => {
  // Initially log the first page view
  logPageView()

  // Log every subsequent route change as a page view
  navigate.bind(() => {
    logPageView()
  })
})

function logPageView () {
  logEvent(analytics, 'page_view', {
    page_path: window.location.pathname
    // Add any additional parameters if needed
  })
}
</script>
<svelte:head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>finGenie Profile</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
</svelte:head>
<Router>
  <div>
    <Route path="/">
      <Chat2 agent="agent"/>
    </Route>
    <Route path="/prompt">
      <PromptView agent="agent"/>
    </Route>
    <Route path="/educate">
      <Chat2 agent="educate"/>
    </Route>
    <Route path="/advisor">
      <AdvisorView/>
    </Route>
    <Route path="/chat">
      <ProfileMenu />
      <div id="score-mode-panel">
        <ModeSwitch />
        <Score/>
      </div>
      <Chat />
    </Route>
    <Route path="restore-conversation">
      <ConversationList />
      <div id="score-mode-panel">
        <ModeSwitch />
        <Score />
      </div>
      <ConversationContent />
    </Route>
    <Route path="profile-form/*">
      <Route path="/">
        <ProfileForm1/>
      </Route>
      <Route path="/2">
        <ProfileForm2/>
      </Route>
      <Route path="/3">
        <ProfileForm3/>
      </Route>
      <Route path="/4">
        <ProfileForm4/>
      </Route>
    </Route>
  </div>
</Router>
