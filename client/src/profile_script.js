import { clearHistory } from './chat_script'
/* eslint-env browser */

export function loadProfiles () {
  const xmlhttp = new XMLHttpRequest()
  xmlhttp.open('GET', '/profiles', false)
  xmlhttp.send()
  if (xmlhttp.status === 200) {
    return JSON.parse(xmlhttp.responseText)
  }
  return null
}

export function generateUserProfileSummary () {
  const newProfile = localStorage.getItem('userData')
  let userProfile
  if (newProfile) {
    const myHeaders = new Headers()
    myHeaders.append('Content-Type', 'application/json')
    myHeaders.append('Accept', 'application/json')

    fetch('/summary', {
      method: 'POST',
      headers: myHeaders,
      body: JSON.stringify({ profile: newProfile }),
      redirect: 'follow'
    })
      .then(response => response.json())
      .then(data => {
        console.log('Response: ', data)
        localStorage.setItem('userSummary', data)
        console.log('Local Storage: ', localStorage)
        userProfile = data
        document.getElementById('user-profile').classList.remove('inactive')
        document.getElementById('clear-user-history').classList.remove('inactive')
      })
  } else {
    userProfile = ''
  }
  return userProfile
}

export function handleMouseOut () {
  const summaryDiv = document.getElementById('summary')
  summaryDiv.removeChild(summaryDiv.firstChild)
  const profiles = document.querySelectorAll('.profile')
  profiles.forEach(function (profile) {
    if (profile.classList.contains('clicked')) showProfileSummary(profile.id)
  })
}

export function handleClick (event) {
  const profiles = document.querySelectorAll('.profile')
  profiles.forEach(function (profile) {
    if (profile.id !== event.target.id) profile.classList.remove('clicked')
  })
  event.target.classList.toggle('clicked')
  showProfileSummary(event.target.id)
  clearHistory(localStorage.getItem('user'))
  localStorage.setItem('user', JSON.stringify({ id: event.target.id, name: event.target.innerHTML }))
}

export function showProfileSummary (id) {
  const summary = document.createElement('p')
  summary.style.color = '#9FACBA'
  const userProfile = localStorage.getItem('userSummary')
  switch (id) {
    case 'profile-0': summary.innerHTML = 'Juan Martinez, a 27-year-old married individual, has an annual income of $375,000 from two salaries, $21,000 in savings and checking accounts, $100,000 in a 401(k), $15,000 in home equity, and $83,000 in debt from credit cards, car payments, and a mortgage.'
      break
    case 'profile-1': summary.innerHTML = 'Jane Doe, a 33-year-old employed single mother with an annual income of $75,000, has $50,000 in student loans and $1,500 in credit card debt, but also has $40,000 in savings and retirement accounts and $5,000 in a college savings account for her two children.'
      break
    case 'profile-2': summary.innerHTML = 'John Doe, a 67-year-old retired individual, has an annual income of $37,000 from pension, social security, and side hustle, while possessing assets worth $1,325,000, including $450,000 in retirement accounts, and owes $11,000 in debts, including $1,000 in credit card debt and a $10,000 mortgage.'
      break
    case 'profile-3': summary.innerHTML = 'Emily Smith, a 21-year-old employed individual, has an annual income of $78,000 and possesses assets worth $4,700, while also carrying debts of $47,000, including a credit card balance of $2,000.'
      break
    default:
      if (userProfile) {
        summary.innerHTML = userProfile
      } else {
        summary.innerHTML = generateUserProfileSummary()
      }
      break
  }

  const summaryDiv = document.getElementById('summary')
  if (summaryDiv.firstChild) {
    summaryDiv.removeChild(summaryDiv.firstChild)
  }
  summaryDiv.appendChild(summary)
}

export function clearUserProfile () {
  localStorage.setItem('userSummary', '')
  localStorage.setItem('userData', '')
  document.getElementById('user-profile').classList.add('inactive')
  document.getElementById('clear-user-history').classList.add('inactive')
}
