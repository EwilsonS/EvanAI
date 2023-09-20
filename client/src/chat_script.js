import { showProfileSummary } from './profile_script'
/* eslint-env browser */

// Initialize an array to store the conversation history
let conversationHistory = []
// global variable to hold the timeout for the conversation
let internalConversationTimeout = null
let internalHistoryTimeout = null
// timeout before new topics are suggested
const conversationTimeout = 30000
// inactivity timeout before conversation history is saved
const historyTimeout = 180000
// state tracking for the suggestion
let suggestionActive = false
let historyActive = false

// Function to load the conversation history from localStorage
export function loadConversationHistory () {
  const storedHistory = localStorage.getItem('conversationHistory')
  const user = JSON.parse(localStorage.getItem('user'))
  const scores = JSON.parse(localStorage.getItem('scores'))
  if (user) {
    const profile = document.getElementById(user.id)
    profile.classList.add('clicked')
    showProfileSummary(user.id)
  }
  if (storedHistory) {
    conversationHistory = JSON.parse(storedHistory)
    conversationHistory.forEach(message => {
      addMessage(message.content, message.role === 'user' ? 'user' : 'bot')
    })
  } else {
    // Display the welcome message if there's no stored history
    displayWelcomeMessage()
  }
  if (scores) {
    displayScore(scores)
  }
}

export function clearHistory (user) {
  // Clear the chat container
  const chatContainer = document.getElementById('chat-container')
  chatContainer.innerHTML = ''

  saveConversation(user)

  // Clear the conversation history array
  conversationHistory = []

  // Clear the conversation history from localStorage
  localStorage.removeItem('conversationHistory')
  localStorage.removeItem('scores')
  // Clear the conversation hsitory from written text file
  writeConversationHistory()
  // Display the welcome message again
  displayWelcomeMessage()

  document.getElementById('response-time-score').style.visibility = 'hidden'
  document.getElementById('csat-score').style.visibility = 'hidden'
  document.getElementById('total-score').style.visibility = 'hidden'
  document.getElementById('ces-score').style.visibility = 'hidden'
  document.getElementById('nps-score').style.visibility = 'hidden'
  document.getElementById('response-quality-score').style.visibility = 'hidden'
}

// Function to add a message to the chat container
export function addMessage (message, sender) {
  const chatContainer = document.getElementById('chat-container')
  const messageElement = document.createElement('div')
  messageElement.className = `message ${sender}-message`

  const messageText = document.createElement('div')
  messageText.className = 'message-text'
  // Apply color class to bot messages based on current mode
  if (sender === 'bot') {
    const color = document.getElementsByTagName('body')[0].style.backgroundColor
    const isDarkMode = color === 'rgb(28, 33, 39)'
    messageText.classList.toggle('dark-mode', isDarkMode)
  }
  messageText.innerHTML = message.replace(/\n/g, '<br>')

  messageElement.appendChild(messageText)
  chatContainer.appendChild(messageElement)

  // Scroll to the bottom of the chat container
  chatContainer.scrollTop = chatContainer.scrollHeight
}

// Event listener for the form submission
export function sendMessage (event) {
  event.preventDefault()
  const input = document.getElementById('message')
  if (input instanceof HTMLInputElement) {
    const messageContent = input.value.trim()

    // Do not proceed if the message is empty
    if (messageContent.length === 0) {
      return
    }

    // Display the user's message and update the conversation history
    addMessage(messageContent, 'user')
    conversationHistory.push({ role: 'user', content: messageContent })
    saveConversationHistory() // Save the updated conversation history to localStorage

    input.value = ''
    input.focus()

    // Display the "Generating response..." message
    displayGeneratingMessage()
    const profiles = document.querySelectorAll('.profile')
    const choosenProfile = Array.from(profiles).filter(p => p.classList.contains('clicked'))
    let profileId = ''
    let userProfile = ''
    if (choosenProfile.length !== 0) {
      profileId = choosenProfile[0].id
      if (profileId === 'user-profile') {
        userProfile = localStorage.getItem('userData')
      }
    }
    // Send a request to the server to generate a response
    fetch('/generate', {
      method: 'POST',
      body: JSON.stringify({ message: messageContent, history: conversationHistory, profile_id: profileId, user_profile: userProfile }),
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json'
      }
    })
      .then(response => response.json())
      .then(data => {
        // Remove the "Generating response..." message and display the bot's response
        removeGeneratingMessage()
        addMessage(data.message.toString(), 'bot')
        setTimeouts()
        conversationHistory.push({ role: 'assistant', content: data.message.toString(), response_time: data.response_time })
        saveConversationHistory() // Save the updated conversation history to localStorage
        console.log(JSON.stringify({ history: conversationHistory }))
        fetch('/score', {
          method: 'POST',
          body: JSON.stringify({ history: conversationHistory }),
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json'
          }
        })
          .then(response => response.json())
          .then(convEval => {
            const respTimeVal = document.getElementById('response-time-score')
            respTimeVal.innerHTML = (convEval.response_time_score + 1).toString() + '/10'
            respTimeVal.style.visibility = 'visible'
            let totalScore = convEval.response_time_score + 1
            let metricCount = 1
            if (convEval.csat_score !== -1) {
              const metric = document.getElementById('csat-score')
              metric.innerHTML = (convEval.csat_score + 1).toString() + '/10'
              metric.style.visibility = 'visible'
              totalScore += convEval.csat_score + 1
              metricCount += 1
            }
            if (convEval.nps_score !== -1) {
              const metric = document.getElementById('nps-score')
              metric.innerHTML = (convEval.nps_score + 1).toString() + '/10'
              metric.style.visibility = 'visible'
              totalScore += convEval.nps_score + 1
              metricCount += 1
            }
            if (convEval.ces_score !== -1) {
              const metric = document.getElementById('ces-score')
              metric.innerHTML = (convEval.ces_score + 1).toString() + '/10'
              metric.style.visibility = 'visible'
              totalScore += convEval.ces_score + 1
              metricCount += 1
            }
            if (convEval.response_quality !== -1) {
              const metric = document.getElementById('response-quality-score')
              metric.innerHTML = (convEval.response_quality + 1).toString() + '/10'
              metric.style.visibility = 'visible'
              totalScore += convEval.response_quality + 1
              metricCount += 1
            }
            const totalVal = document.getElementById('total-score')
            totalVal.innerHTML = (Math.round((totalScore) / metricCount * 10) / 10).toString() + '/10'
            totalVal.style.visibility = 'visible'
          })
      })
  }
  // Update the suggestion active flag
  suggestionActive = false
  historyActive = false
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

export function clearTimeouts () {
  // Update the suggestion active flag
  if (internalConversationTimeout) {
    clearTimeout(internalConversationTimeout)
  }
  if (internalHistoryTimeout) {
    clearTimeout(internalHistoryTimeout)
  }
}

export function displayScore (score) {
  document.getElementById('response-time-score').style.visibility = 'visible'
  document.getElementById('csat-score').style.visibility = 'visible'
  document.getElementById('total-score').style.visibility = 'visible'
  document.getElementById('ces-score').style.visibility = 'visible'
  document.getElementById('nps-score').style.visibility = 'visible'
  document.getElementById('response-quality-score').style.visibility = 'visible'

  document.getElementById('response-time-score').innerHTML = score.response_time_score
  document.getElementById('csat-score').innerHTML = score.csat_score
  document.getElementById('total-score').innerHTML = score.total_score
  document.getElementById('ces-score').innerHTML = score.ces_score
  document.getElementById('nps-score').innerHTML = score.nps_score
  document.getElementById('response-quality-score').innerHTML = score.response_quality_score
}

function generateInternalConversation () {
  if (suggestionActive) {
    return
  }
  const message = `Based on our conversation, state a list of three questions\n
        I could ask YOU next? Format it as a bulleted list and start it\n
        with 'Here are some followup questions that may interest you:'`

  conversationHistory.push({ role: 'user', content: message })

  fetch('/generate', {
    method: 'POST',
    body: JSON.stringify({ message, history: conversationHistory }),
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json'
    }
  })
    .then(response => response.json())
    .then(data => {
      conversationHistory.pop()
      conversationHistory.push({ role: 'assistant', content: data.message })
      saveConversationHistory()
      addMessage(data.message, 'bot')
    })
  suggestionActive = true
}

function writeConversationHistory () {
  if (historyActive) return
  if (conversationHistory.length <= 0) return
  // Send a request to the server to generate a response

  const raw = JSON.stringify({ history: conversationHistory })
  console.log(raw)
  fetch('/save_history', {
    method: 'POST',
    body: raw,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json'
    }
  })
    .then(response => response.json())
    .then(data => {
      // Remove the "Generating response..." message and display the bot's response
      console.log(data)
    })

  historyActive = true
}

function setTimeouts () {
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

// Function to display the welcome message from the bot
function displayWelcomeMessage () {
  const welcomeMessage = `Hello! I'm finGenie. How can I help you today?

    Here are some example questions:
    - How do I create a budget?
    - What are some strategies for paying off debt?
    - How much should I be saving for retirement?
    - What is a credit score and why is it important?
    - How can I invest my money wisely?
    - What are some ways to save money on everyday expenses?
    - How can I track my spending and stay within my budget?
    - What should I consider when choosing a credit card or bank account?
    - How can I improve my financial literacy?
    - What are some common mistakes people make with their finances?
    `
  addMessage(welcomeMessage, 'bot')
  setTimeouts()
}

// Function to display a temporary "Generating response..." message from the bot
function displayGeneratingMessage () {
  const generatingMessage = 'Generating response...'
  addMessage(generatingMessage, 'bot')
  return generatingMessage
}

// Function to remove the "Generating response..." message from the chat container
function removeGeneratingMessage () {
  const chatContainer = document.getElementById('chat-container')
  chatContainer.removeChild(chatContainer.lastChild)
}

function saveConversation (user) {
  // Save conversation to continue later if needed
  if (user === null || user === undefined) user = localStorage.getItem('user')
  const savedConversations = localStorage.getItem('savedConversations')
  const score = {
    response_time_score: document.getElementById('response-time-score').innerHTML,
    csat_score: document.getElementById('csat-score').innerHTML,
    total_score: document.getElementById('total-score').innerHTML,
    ces_score: document.getElementById('ces-score').innerHTML,
    nps_score: document.getElementById('nps-score').innerHTML,
    response_quality_score: document.getElementById('response-quality-score').innerHTML
  }
  if (savedConversations) {
    const history = JSON.parse(savedConversations)
    history.push({ user: JSON.parse(user), history: conversationHistory, scores: score })
    localStorage.setItem('savedConversations', JSON.stringify(history))
  } else {
    localStorage.setItem('savedConversations', JSON.stringify([{ user: JSON.parse(user), history: conversationHistory, scores: score }]))
  }
}

// Function to save the conversation history to localStorage
function saveConversationHistory () {
  localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory))
  const profiles = document.querySelectorAll('.profile')
  profiles.forEach(function (el) {
    if (el.classList.contains('clicked')) localStorage.setItem('user', JSON.stringify({ id: el.id, name: el.innerHTML }))
  })
}
