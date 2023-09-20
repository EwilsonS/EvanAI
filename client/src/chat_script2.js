/* eslint-env browser */
import { chosenProfile } from './lib/stores'
import UserFeedback from './lib/components/UserFeedback.svelte'
import ChatMessage from './lib/components/ChatMessage.svelte'
import Icon from './lib/components/Icon.svelte'
import { logEvent } from 'firebase/analytics'
import analytics from './firebase.js'
import { langOpts } from './switch_language'

export let baseUrl = import.meta.env.VITE_FIN_GENIE_BASE_URL
if (baseUrl === undefined) {
  baseUrl = 'http://localhost:8080'
}

let welcomeMessage = ''
let agent = ''
const lang = localStorage.getItem('display_language') || 'en'

function setAgentAndWelcomeMsg () {
  agent = new URLSearchParams(window.location.search).get('agent')
  welcomeMessage = langOpts.welcome[lang]
  if (agent === 'educate') {
    welcomeMessage = langOpts.welcomeEducate[lang]
  }
}
// global variable to hold the timeout for the conversation
// Initialize an array to store the conversation history
let conversationHistory = []
let internalConversationTimeout
let internalHistoryTimeout
const nudgeTimeout = 3 * 60 * 1000 // 3 minutes before nudge
let nudge // used to track the nudge callback
let nudgeActive = false // do not nudge the user if they are already active
// timeout before new topics are suggested
const conversationTimeout = 1 * 60 * 1000 // 1 minute before new suggestions
// inactivity timeout before conversation history is saved
const historyTimeout = 180000
// state tracking for the suggestion
const maxSuggestions = 2 // maximum number of suggestions to display
let suggestionCount = 0 // number of suggestions displayed
let suggestionActive = false // flag to prevent multiple suggestions
let historyActive = false
let profile = ''
chosenProfile.subscribe((value) => {
  if (Object.keys(value).includes('id')) {
    const pattern = /\w+-(\d+)/
    const id = value.id.match(pattern)[1]
    profile = id
  }
})

// let userProfile = localStorage.getItem('userSummary') // eslint-disable-line no-undef
let messageIndex = 0
const messageClassIndex = 1

// analytics events
const eventUserInput = 'user_input'
const eventAgentSuggestion = 'agent_suggestion'
const eventClearHistory = 'clear_history'
const eventCopyMessage = 'copy_message'
const eventCopyConversation = 'copy_conversation'
const eventCopySummary = 'copy_summary'
export const eventHamburgerMenu = 'hamburger_menu'
const eventScrollUp = 'scroll_up'
const eventClickLink = 'click_link'
const eventLogin = 'login'
const eventFeedback = 'user_feedback'
const eventClickSuggestion = 'click_suggestion'

let username = localStorage.getItem('username')
let password = localStorage.getItem('password')

if (!username || !password) {
  username = prompt('Please enter your username:')
  password = prompt('Please enter your password:')

  if (username && password) {
    trackAnalyticsEvent(eventLogin, { method: 'prompt' })
    localStorage.setItem('username', username)
    localStorage.setItem('password', password)
  }
}
export const creds = btoa(username + ':' + password)

// Function to load the conversation history from localStorage
export function loadConversationHistory () {
  const storedHistory = localStorage.getItem('conversationHistory') // eslint-disable-line no-undef
  if (storedHistory) {
    conversationHistory = JSON.parse(storedHistory)
    conversationHistory.forEach(message => {
      addMessage(message.content, message.role === 'user' ? 'user' : 'bot')
    })
    loadFeedback()
  } else {
    // Display the welcome message if there's no stored history
    setAgentAndWelcomeMsg()
    displayWelcomeMessage()
  }
}

export function clearHistory () {
  trackAnalyticsEvent(eventClearHistory)
  suggestionActive = false
  suggestionCount = 0
  nudgeActive = false // do not nudge the user after clearing the history
  // Clear the chat container
  const chatContainer = document.getElementById('chat-container2')
  chatContainer.innerHTML = ''

  // Clear the conversation history array
  conversationHistory = []

  // Clear the messageIndex for copy functionality
  messageIndex = 0
  // Clear the conversation history from localStorage
  localStorage.removeItem('conversationHistory') // eslint-disable-line no-undef
  // Clear the conversation hsitory from written text file
  writeConversationHistory()
  // Display the welcome message again
  displayWelcomeMessage()
  changeInputFocus() // set focus to text input box
}

// Function to save the conversation history to localStorage
export function saveConversationHistory () {
  localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory)) // eslint-disable-line no-undef
  const profiles = document.querySelectorAll('.profile')
  profiles.forEach(function (el) {
    if (el.classList.contains('clicked')) localStorage.setItem('user', el.id) // eslint-disable-line no-undef
  })
}

export function handleEnter (event) {
  if (event.key === 'Enter') {
    sendMessage(undefined) // Call the function you want to run on Enter key press
  }
}

function addStreamedMessage (chunk, sender, chunkSpan) {
  const chatContainer = document.getElementById('chat-container2')
  // let copyBtn
  let userButtons
  let messageContainer
  let innerMessageContainer
  let chunkDiv
  let icon

  // message div
  if (chunkSpan === undefined) {
    // Copy button
    userButtons = document.createElement('div')

    // remove spacing element from previous message
    let spacingDiv
    if (document.getElementById('spacing')) {
      spacingDiv = document.getElementById('spacing')
      chatContainer.removeChild(spacingDiv)
    }
    // outer div container
    messageContainer = document.createElement('div')
    messageContainer.id = `${sender}-message-container`

    // inner message container
    innerMessageContainer = document.createElement('div')
    innerMessageContainer.className = 'message-container'

    // inner message icon
    icon = new Icon({
      target: innerMessageContainer,
      props: {
        sender
      }
    })
    // inner message component
    chunkDiv = new ChatMessage({
      target: innerMessageContainer,
      props: {
        sender,
        index: messageIndex,
        animate: false,
        message: '',
        callback: undefined,
        addMessage: false
      }
    })

    chunkSpan = chunkDiv.getChunkSpan()
  }

  chunkSpan.innerHTML += chunk.replace(/\n/g, '<br>')
  buttonizeSuggestions(chunkSpan)
  if (userButtons && messageContainer && innerMessageContainer && icon) {
    // Append user feedback buttons component
    userButtons = new UserFeedback({
      target: innerMessageContainer,
      props: {
        feedbackActor: sender,
        componentIndex: messageIndex,
        activity: 'inactive'
      }
    })

    // spacing at bottom of the page
    const messageSpace = document.createElement('div')
    messageSpace.id = 'spacing'
    messageSpace.className = 'message-spacing'
    messageSpace.innerHTML = '<br><br><br>'
    messageContainer.append(innerMessageContainer)
    chatContainer.append(messageContainer)
    chatContainer.append(messageSpace)
    messageIndex++
  }
  // Scroll to the bottom of the chat container
  chatContainer.scrollTop = chatContainer.scrollHeight

  // Clear any existing internal conversation timeout
  if (internalConversationTimeout) {
    clearTimeout(internalConversationTimeout)
  }
  if (internalHistoryTimeout) {
    clearTimeout(internalHistoryTimeout)
  }
  // Set a new internal conversation timeout
  internalConversationTimeout = setTimeout(generateInternalConversation, conversationTimeout)
  internalHistoryTimeout = setTimeout(writeConversationHistory, historyTimeout)

  return chunkSpan
}

export function addMessage (message, sender, animate = false, callback = undefined) {
  const chatContainer = document.getElementById('chat-container2')
  // remove spacing element from previous message
  let spacingDiv
  if (document.getElementById('spacing')) {
    spacingDiv = document.getElementById('spacing')
    chatContainer.removeChild(spacingDiv)
  }
  // outer div container
  const messageContainer = document.createElement('div')
  messageContainer.id = `${sender}-message-container`
  // inner message container
  const innerMessageContainer = document.createElement('div')
  innerMessageContainer.className = 'message-container'

  // spacing at bottom of the page
  const messageSpace = document.createElement('div')
  messageSpace.id = 'spacing'
  messageSpace.className = 'message-spacing'
  messageSpace.innerHTML = '<br><br><br>'

  // User Icon component
  const icon = new Icon({
    target: innerMessageContainer,
    props: {
      sender
    }
  })
  console.log('Icon: ' + icon)
  // Chat Message Component
  const messageDiv = new ChatMessage({
    target: innerMessageContainer,
    props: {
      sender,
      index: messageIndex,
      animate,
      message,
      callback,
      addMessage: true
    }
  })
  console.log('MessageDiv: ' + messageDiv)
  if (sender === 'bot' && message !== welcomeMessage) {
    // User Feedback buttons component
    const userButtons = new UserFeedback({
      target: innerMessageContainer,
      props: {
        feedbackActor: sender,
        componentIndex: messageIndex,
        activity: 'active'
      }
    })
    console.log('User Feedback Buttons: ' + userButtons)
  }

  // Mount elements to page
  messageContainer.append(innerMessageContainer)
  chatContainer.append(messageContainer)
  chatContainer.append(messageSpace)

  // Do not count welcome message in index
  if (message !== welcomeMessage) messageIndex++

  // Scroll to the bottom of the chat container
  chatContainer.scrollTop = chatContainer.scrollHeight

  // Clear any existing internal conversation timeout
  if (internalConversationTimeout) {
    clearTimeout(internalConversationTimeout)
  }
  if (internalHistoryTimeout) {
    clearTimeout(internalHistoryTimeout)
  }
  // Set a new internal conversation timeout
  internalConversationTimeout = setTimeout(generateInternalConversation, conversationTimeout)
  internalHistoryTimeout = setTimeout(writeConversationHistory, historyTimeout)
}

export function copyMessage (event) {
  trackAnalyticsEvent(eventCopyMessage)
  // Retrieve index of the clicked copy button
  const index = event.target.classList[messageClassIndex]

  // Retrieve the corresponding message
  const messageToCopy = document.getElementById(`${index}`)

  if (messageToCopy) {
    const textToCopy = messageToCopy.innerHTML.replaceAll('\n', '<br>')
    try {
      navigator.clipboard.write([new ClipboardItem({ 'text/html': textToCopy })])
      console.log('Copied to clipboard!')
      event.target.innerHTML += '<br> done'
      event.target.style.color = 'green'
      setTimeout(() => {
        event.target.innerHTML = 'content_copy'
        event.target.style.color = 'lightgray'
      }, 500)
    } catch (err) {
      console.error('Unable to copy', err)
    }
  }
}

export function setFeedback (event, index, value, oppositeButton, oppositeSymbol) {
  trackAnalyticsEvent(eventFeedback, { satisfied: value })
  conversationHistory.at(index).satisfied = value
  localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory))
  event.target.innerHTML += '<br> done'
  if (value) event.target.style.color = 'green'
  else event.target.style.color = 'red'
  oppositeButton.style.color = 'lightgray'
  oppositeButton.innerHTML = oppositeSymbol
}

function loadFeedback () {
  let userFeedbackButton
  let feedbackValue
  for (let i = 0; i < conversationHistory.length; i++) {
    if (typeof (conversationHistory.at(i).satisfied) !== 'undefined' && conversationHistory.at(i).role === 'assistant') {
      console.log('Conversation History Element ' + i + ': ' + JSON.stringify(conversationHistory.at(i)))
      feedbackValue = conversationHistory.at(i).satisfied

      switch (feedbackValue) {
        case true:
          userFeedbackButton = document.getElementById(`bot-up-${i}`)
          userFeedbackButton.style.color = 'green'
          userFeedbackButton.innerHTML += '<br> done'
          break

        case false:
          userFeedbackButton = document.getElementById(`bot-down-${i}`)
          userFeedbackButton.style.color = 'red'
          userFeedbackButton.innerHTML += '<br> done'
          break

        default:
          break
      }
    }
  }
}

export function copyConversation (event) {
  trackAnalyticsEvent(eventCopyConversation)
  let resultFinal = ''
  conversationHistory.forEach((item) => {
    const result1 = item.role + ''
    const result2 = item.content
    resultFinal = '<br>' + resultFinal + result1.toUpperCase() + ':' + result2 + '<br><br>'
    resultFinal = resultFinal.replaceAll('\n', '<br>')
  }
  )
  event.target.innerHTML += ' done'
  event.target.style.color = 'green'
  setTimeout(() => {
    event.target.innerHTML = 'content_copy'
    event.target.style.color = 'lightgray'
  }, 500)
  navigator.clipboard.write([new ClipboardItem({ 'text/html': `${resultFinal}` })])
}

export function clearBlinkers () {
  const blinkerSpans = document.getElementsByClassName('blinker')

  Array.from(blinkerSpans).forEach(span => {
    span.style.animation = 'fadeout 1s 0s forwards'
    setTimeout(() => {
      // span.style.animation = 'fadeout 1s 0s reverse'
      span.remove()
    }, 2000)
  })
}

function setTimeouts (delay = conversationTimeout) {
  // const delay = conversationTimeout // * Math.pow(2, suggestionCount)

  if (internalConversationTimeout) { clearTimeout(internalConversationTimeout) }
  internalConversationTimeout = setTimeout(generateInternalConversation, delay)

  if (internalHistoryTimeout) { clearTimeout(internalHistoryTimeout) }
  // save the conversation history after a delay
  internalHistoryTimeout = setTimeout(writeConversationHistory, historyTimeout)

  if (nudge) { clearTimeout(nudge) }
  nudge = setTimeout(nudgeUser, nudgeTimeout) // nudge the user after inactivity
}

export function displayWelcomeMessage () {
  const callback = () => { setTimeout(generateInternalConversation, 500) }
  addMessage(welcomeMessage, 'bot', true, callback)
}

export function nudgeUser () {
  /**
   * Nudge the user to continue the conversation.
   */
  if (nudgeActive === false) {
    addMessage(langOpts.nudge[lang], 'bot', true)
    nudgeActive = true // do not spam the user with nudges
    suggestionActive = true // do not suggest new topics after nudging
  }
}

function getProfileId () {
  /**
   * Get the selected profile.
   */
  // this impl will have problems if we start to use more than one param in the url.
  let profileId = window.location.search.substr(1).split('=')[1] // grab the profile id from the url
  if (profile !== '') profileId = profile
  return profileId
}

// Event listener for the form submission
export function sendMessage (event) {
  event.preventDefault()
  const input = document.getElementById('message')
  if (input instanceof HTMLInputElement) { // eslint-disable-line no-undef
    const messageContent = input.value.trim()

    // Do not proceed if the messageContent is empty
    if (messageContent.length === 0) {
      return
    }

    // Display the user's messageContent and update the conversation history
    addMessage(messageContent, 'user')
    trackAnalyticsEvent(eventUserInput)
    conversationHistory.push({ role: 'user', content: messageContent })
    saveConversationHistory() // Save the updated conversation history to localStorage

    input.value = '' // Clear the input field
    input.focus() // set focus to text input box

    const profiles = document.querySelectorAll('.profile')
    const chosenProfile = Array.from(profiles).filter(p => p.classList.contains('clicked'))
    let profileId = getProfileId()
    let userProfile = ''
    if (chosenProfile.length !== 0) {
      profileId = chosenProfile[0].id
      if (profileId === 'user-profile') {
        userProfile = localStorage.getItem('userProfile') // eslint-disable-line no-undef
      }
    }
    // Send a request to the server to generate a response
    let chunkElement = addStreamedMessage('', 'bot', undefined) // make the cursor visible
    fetch(baseUrl + '/generate-stream', {
      method: 'POST',
      body: JSON.stringify({ message: messageContent, history: conversationHistory, profile_id: profileId, user_profile: userProfile, agent, language: lang }),
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Basic ' + creds,
        Accept: 'text/event-stream'
      }
    })
      .then(response => {
        const reader = response.body.getReader()
        const textDecoder = new TextDecoder()

        const accumulatedText = []
        return reader.read().then(function processText ({ done, value }) {
          if (done) { // stream complete
            if (accumulatedText.length > 0) {
              setTimeout(() => clearBlinkers(), 1000) // clear cursors after a brief delay
              const links = chunkElement.querySelectorAll('a') // add event listeners to links
              links.forEach(link => {
                link.addEventListener('click', function (event) {
                  const targetText = event.target.textContent || event.target.innerText
                  trackAnalyticsEvent(eventClickLink, { text: targetText })
                })
              })
              conversationHistory.push({ role: 'assistant', content: accumulatedText.join('') })
              saveConversationHistory() // Save the updated conversation history to localStorage
              setTimeouts()
            }
            return
          }

          // Decode the chunk of text and do something with it
          const chunk = textDecoder.decode(value)
          accumulatedText.push(chunk)
          chunkElement = addStreamedMessage(chunk.toString(), 'bot', chunkElement)

          // Read the next chunk
          return reader.read().then(processText)
        })
      })
      .catch(error => {
        console.error('An error occurred:', error)
      })
      .finally(() => {
        const userButtons = document.getElementById(`button-container-${messageIndex - 1}`)
        userButtons.classList.remove('inactive')
        userButtons.classList.add('active')
      })
  }
  // Update the suggestion active flag
  suggestionActive = false
  nudgeActive = false
  historyActive = false
  changeInputFocus(event) // set focus to text input box
}

export function generateInternalConversation () {
  if (suggestionActive || nudgeActive || suggestionCount >= maxSuggestions) {
    return
  }
  const prompt = `State a list of three questions I should ask next.\n
  Ask the questions from the first person perspective.
  All questions need to be a single quesion each and nothing else.
  Respond in the ${langOpts.countryCode[lang]} language only.
  Format it as a bulleted list and start it with '${langOpts.suggestionsPrompt[lang]}:'`

  const profileId = getProfileId()

  conversationHistory.push({ role: 'user', content: prompt })

  fetch(baseUrl + '/generate-stream', {
    method: 'POST',
    body: JSON.stringify({ prompt, history: conversationHistory, profile_id: profileId, agent, language: lang }),
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Basic ' + creds,
      Accept: 'text/event-stream'
    }
  })
    .then(response => {
      const reader = response.body.getReader()
      const textDecoder = new TextDecoder()

      const accumulatedText = []
      let chunkElement
      return reader.read().then(function processText ({ done, value }) {
        if (done) { // stream complete
          trackAnalyticsEvent(eventAgentSuggestion)
          setTimeout(() => clearBlinkers(), 1000) // clear cursors after a brief delay
          if (accumulatedText.length > 0) {
            const links = chunkElement.querySelectorAll('a') // add event listeners to links
            links.forEach(link => {
              link.addEventListener('click', function (event) {
                const targetText = event.target.textContent || event.target.innerText
                trackAnalyticsEvent(eventClickLink, { text: targetText })
              })
            })
            conversationHistory.pop() // remove the last prompt
            conversationHistory.push({ role: 'assistant', content: accumulatedText.join(''), satisfied: undefined })
            saveConversationHistory() // Save the updated conversation history to localStorage
            suggestionActive = true
            suggestionCount++
            setTimeouts()
          }
          return
        }

        // Decode the chunk of text and do something with it
        const chunk = textDecoder.decode(value)
        accumulatedText.push(chunk)
        chunkElement = addStreamedMessage(chunk.toString(), 'bot', chunkElement)

        // Read the next chunk
        return reader.read().then(processText)
      })
    })
    .catch(error => {
      console.error('An error occurred:', error)
    })
    .finally(() => {
      const userButtons = document.getElementById(`button-container-${messageIndex - 1}`)
      userButtons.classList.remove('inactive')
      userButtons.classList.add('active')
    })
}

export function generateConversationSummary () {
  const summaryContainer = document.getElementById('popup-summary-text')
  summaryContainer.innerHTML = ''
  const botResponses = []
  conversationHistory.forEach(element => {
    if (element.role === 'assistant') {
      botResponses.push(element)
    }
  })
  const prompt = `${langOpts.summaryPrompt[lang]}. Respond in the ${lang} language.` // best response so far
  botResponses.push({ role: 'user', content: prompt })

  fetch(baseUrl + '/generate-stream', {
    method: 'POST',
    body: JSON.stringify({ prompt, history: botResponses, agent, language: lang }),
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Basic ' + creds,
      Accept: 'application/json'
    }
  })
    .then(response => {
      const reader = response.body.getReader()
      const textDecoder = new TextDecoder()

      const accumulatedText = []
      return reader.read().then(function processText ({ done, value }) {
        if (done) {
          trackAnalyticsEvent(eventAgentSuggestion)
          setTimeout(() => clearBlinkers(), 1000)
          if (accumulatedText.length > 0) {
            conversationHistory.pop()
          }
          return
        }
        const chunk = textDecoder.decode(value)
        accumulatedText.push(chunk)
        console.log(chunk.toString())
        addSummary(chunk.toString())
        return reader.read().then(processText)
      })
    })
    .catch(error => {
      console.error('An error occured: ', error)
    })
}

function addSummary (summary) {
  const summaryElement = document.getElementById('popup-summary-text')
  summaryElement.innerHTML += summary.replace(/\n/g, '<br>')
  const links = summaryElement.querySelectorAll('a') // add event listeners to links
  links.forEach(link => {
    link.addEventListener('click', function (event) {
      const targetText = event.target.textContent || event.target.innerText
      trackAnalyticsEvent(eventClickLink, { text: targetText })
    })
  })
  summaryElement.scrollTop = summaryElement.scrollHeight
}

export function copySummary (event) {
  trackAnalyticsEvent(eventCopySummary)
  const summaryElement = document.getElementById('popup-summary-text')
  navigator.clipboard.write([new ClipboardItem({ 'text/html': summaryElement.innerHTML.replaceAll('\n', '<br>') })])
  event.target.innerHTML += ' done'
  event.target.style.color = 'green'
  setTimeout(() => {
    event.target.innerHTML = 'content_copy'
    event.target.style.color = 'lightgray'
  }, 500)
}

export function clearSuggestions () {
  // Clear any existing internal conversation timeout
  if (internalConversationTimeout) {
    clearTimeout(internalConversationTimeout)
  }
  if (internalHistoryTimeout) {
    clearTimeout(internalHistoryTimeout)
  }
  internalConversationTimeout = setTimeout(generateInternalConversation, conversationTimeout)
  internalHistoryTimeout = setTimeout(writeConversationHistory, historyTimeout)
}

export function writeConversationHistory () {
  if (historyActive) return
  if (conversationHistory.length <= 0) return
  // Send a request to the server to generate a response

  const raw = JSON.stringify({ history: conversationHistory })
  fetch(baseUrl + '/save_history', {
    method: 'POST',
    body: raw,
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Basic ' + creds,
      Accept: 'application/json'
    }
  })
    .then(response => response.json())
    .then(data => { // TODO if we are not using data remove it
      // Remove the "Generating response..." message and display the bot's response
      console.log(data)
    })

  historyActive = true
}

function changeInputFocus (event, selector = 'input[name="message"]') {
  /**
   * Change the focus to the input element by querying the selector.
   */
  const inputElement = document.querySelector(selector)
  inputElement.focus()
}

export function trackAnalyticsEvent (event, eventParams = {}) {
  logEvent(analytics, event, eventParams)
}

let scrollTimeout
let lastScrollTop = 0

function addScrollEvent () {
  document.getElementById('chat-container2').addEventListener('scroll', function (event) {
    clearTimeout(scrollTimeout)
    let scrollUpDirection = false
    const currentScrollTop = event.target.scrollTop
    if (currentScrollTop < lastScrollTop) {
      scrollUpDirection = true
    }
    lastScrollTop = currentScrollTop
    scrollTimeout = setTimeout(function () {
      if (scrollUpDirection) { // only track scroll up events
        trackAnalyticsEvent(eventScrollUp)
      }
    }, 500)
  })
}

export function buttonizeSuggestions (chunkSpan) {
  const questionMarksCount = chunkSpan.innerHTML.split('?').length - 1
  if (questionMarksCount === 3 && chunkSpan.innerHTML.trim().includes(langOpts.suggestionsPrompt[lang])) {
    const suggestions = chunkSpan.innerHTML.trim()
    const suggestionMessage = suggestions.split(/:(?![^<]*>)/)[0]
    const suggestionArr = suggestions.split(/:(?![^<]*>)/)[1].split('?')
    chunkSpan.innerHTML = `${suggestionMessage}: <br><br>`
    let additionalMsg = ''

    if (suggestionArr[suggestionArr.length - 1] === '') {
      suggestionArr.pop()
    }

    if (suggestionArr.length !== 3) {
      additionalMsg = suggestionArr[suggestionArr.length - 1]
      suggestionArr.pop()
    }

    suggestionArr.forEach((item, index) => {
      const suggestion = document.createElement('span')
      const sendSuggestionBtn = document.createElement('span')
      const text = `${item.trim().replaceAll('<br>', '')}?`
      suggestion.innerHTML += text
      sendSuggestionBtn.innerHTML += '<i class="send-suggestion-btn material-icons">send</i>'
      if (index !== (suggestionArr.length - 1)) {
        sendSuggestionBtn.innerHTML += '<br>'
      }
      sendSuggestionBtn.onclick = (e) => clickSuggestion(text.slice(2).replace(/<[^>]+>/g, ''), e)
      suggestion.appendChild(sendSuggestionBtn)
      chunkSpan.appendChild(suggestion)
    })

    if (additionalMsg) {
      const additionalMsgSpan = document.createElement('span')
      additionalMsgSpan.innerHTML += `<br><br>${additionalMsg}`
      chunkSpan.appendChild(additionalMsgSpan)
    }
  }
}

function clickSuggestion (text, event) {
  trackAnalyticsEvent(eventClickSuggestion, { text })
  document.getElementById('message').value = text
  sendMessage(event)
}

document.addEventListener('DOMContentLoaded', function () {
  setTimeout(() => { addScrollEvent() }, 0) // add scroll event listener
  // Get the input field by its ID
  const inputField = document.getElementById('message')
  setTimeout(() => { changeInputFocus('DOMContentLoaded') }, 100) // set focus to text input box after 100ms

  // Attach the keyup event listener
  inputField.addEventListener('keyup', function () {
    setTimeouts()
  })

  // Attach the focus event listener
  inputField.addEventListener('focus', function () {
    setTimeouts()
  })
})

export async function getSystemCards () {
  try {
    const response = await fetch(baseUrl + '/get-system-card-list', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Basic ' + creds,
        Accept: 'application/json'
      }
    })
    const data = await response.json()
    return data
  } catch (error) {
    console.log(error)
  }
}

export async function getCardData (name) {
  const fileName = name
  try {
    const response = await fetch(baseUrl + '/read-system-card/' + name, {
      method: 'POST',
      body: JSON.stringify({ file_name: fileName }),
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Basic ' + creds,
        Accept: 'application/json'
      }
    })
    const data = await response.json()
    return data
  } catch (error) {
    console.log(error)
  }
}

export async function saveNewData (name, data) {
  const fileName = name
  const content = data
  try {
    const response = await fetch(baseUrl + '/update-system-card', {
      method: 'PUT',
      body: JSON.stringify({
        file_name: fileName,
        updated_system_card_content: content
      }),
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Basic ' + creds,
        Accept: 'application/json'
      }
    })
    const data = await response.json()
    console.log('Final Response', data)
    return data
  } catch (error) {
    console.log(error)
  }
}
