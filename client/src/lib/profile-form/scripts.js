import Dependents from './fields/dependents.svelte'
import Income from './fields/income.svelte'
import Assets from './fields/assets.svelte'
import Loans from './fields/loans.svelte'

let userData

buildJSON()
function buildJSON () {
  const basicsJson = '{"profile":{"first_name": "", "last_name": "", "age": "", "dependents": [], "marital_status": "", "employment_status": "",'
  const assetsJson = '"annual_income":[], "assets":[],'
  const debtsJson = '"debts": { "credit_card":"", "loans":[] }, '
  const expenseJson = '"monthly_expenses": {"gas":"", "phone":"", "utilities":"", "entertainment":"", "food": "", "childcare": "", "other": ""} }}'
  const fullJson = basicsJson + assetsJson + debtsJson + expenseJson

  userData = JSON.parse(fullJson)
  localStorage.setItem('userData', JSON.stringify(userData)) // eslint-disable-line no-undef
}

// ----PROFILE PAGE 1----
function getAge (dob) {
  const today = new Date()
  const birthDate = new Date(dob)
  let age = today.getFullYear() - birthDate.getFullYear()
  const m = today.getMonth() - birthDate.getMonth()
  if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
    age--
  }
  return age
}

export function addBasics (dob, firstname, lastname, maritalstatus, employment) {
  const age = getAge(dob)

  userData.profile.first_name = firstname.toString()
  userData.profile.last_name = lastname.toString()
  userData.profile.age = age.toString()
  userData.profile.marital_status = maritalstatus.toString()
  userData.profile.employment_status = employment.toString()

  localStorage.setItem('userData', JSON.stringify(userData)) // eslint-disable-line no-undef
}

// ----PROFILE PAGE 3----
export function addExpenses (creditCard, rent, gas, phone, utilities,
  entertainment, food, childcare, other) {
  userData.profile.debts.credit_card = creditCard
  userData.profile.monthly_expenses.rent = rent
  userData.profile.monthly_expenses.gas = gas
  userData.profile.monthly_expenses.phone = phone
  userData.profile.monthly_expenses.utilities = utilities
  userData.profile.monthly_expenses.entertainment = entertainment
  userData.profile.monthly_expenses.food = food
  userData.profile.monthly_expenses.childcare = childcare
  userData.profile.monthly_expenses.other = other

  localStorage.setItem('userData', JSON.stringify(userData)) // eslint-disable-line no-undef
  console.log('USER DATA: ' + JSON.stringify(userData))
}

// ----DEPENDENTS----
let dependentType = ''
let dependentAge = ''
export const dependents = []

// ----ASSETS----
let assetType = ''
let assetAmount = ''
export const assets = []

// ----INCOME----
let incomeType = ''
let incomeAmount = ''
export const incomes = []

// ----LOAN----
let loanType = ''
let loanAmount = ''
let loanPrincipal = ''
export const loans = []

// ----UNIVERSAL----
export function setListType (index, type) {
  switch (index) {
    case 2: // ASSETS
      assetType = type
      break
    case 3: // LOANS
      loanType = type
      break
    case 4: // DEPENDENTS
      dependentType = type
      break
    default: // INCOME
      incomeType = type
      break
  }
}

export function setListAmount (index, amount) {
  switch (index) {
    case 2: // ASSETS
      assetAmount = amount
      break
    case 3: // LOANS
      loanAmount = amount
      break
    case 4: // DEPENDENTS
      dependentAge = amount
      break
    default: // INCOME
      incomeAmount = amount
      break
  }
}

export function setLoanPrincipal (principal) {
  loanPrincipal = principal
}

export function setListObject (index) {
  switch (index) {
    case 2: { // ASSETS
      const assetRaw = '{"type": "", "amount": ""}'
      const assetJSON = JSON.parse(assetRaw)
      assetJSON.type = assetType
      assetJSON.amount = assetAmount

      userData.profile.assets.push(assetJSON)
      break
    }
    case 3: { // LOANS
      const loanRaw = '{"type": "", "amount": "", "principal": ""}'
      const loanJSON = JSON.parse(loanRaw)
      loanJSON.type = loanType
      loanJSON.amount = loanAmount
      loanJSON.principal = loanPrincipal

      userData.profile.debts.loans.push(loanJSON)
      break
    }
    case 4: { // DEPENDENTS
      const dependentRaw = '{"type": "", "age": ""}'
      const dependentJSON = JSON.parse(dependentRaw)
      dependentJSON.type = dependentType
      dependentJSON.age = dependentAge

      userData.profile.dependents.push(dependentJSON)
      break
    }
    default: { // INCOME
      const incomeRaw = '{"type": "", "amount": ""}'
      const incomeJSON = JSON.parse(incomeRaw)
      incomeJSON.type = incomeType
      incomeJSON.amount = incomeAmount

      userData.profile.annual_income.push(incomeJSON)
      break
    }
  }
}

export function addField (index) {
  let fieldName

  switch (index) {
    case 2: {
      fieldName = 'asset'

      const assetFieldContainer = document.getElementById(fieldName)
      const btnID = 'profile-' + fieldName + '-btn'
      const assetFieldBtn = document.getElementById(btnID)
      assetFieldContainer.removeChild(assetFieldBtn)

      const assetNewFieldDiv = new Assets({ target: document.getElementById('asset') })
      console.log(assetNewFieldDiv)
      // setMode()
      break
    }
    case 3: {
      fieldName = 'loan'

      const loanFieldContainer = document.getElementById(fieldName)
      const btnID = 'profile-' + fieldName + '-btn'
      const loanFieldBtn = document.getElementById(btnID)
      loanFieldContainer.removeChild(loanFieldBtn)

      const loanNewFieldDiv = new Loans({ target: document.getElementById('loan') })
      console.log(loanNewFieldDiv)
      // setMode()
      break
    }
    case 4: {
      fieldName = 'dependent'

      const dependentFieldContainer = document.getElementById(fieldName)
      const btnID = 'profile-' + fieldName + '-btn'
      const dependentFieldBtn = document.getElementById(btnID)
      dependentFieldContainer.removeChild(dependentFieldBtn)

      const dependentNewFieldDiv = new Dependents({ target: document.getElementById('dependent') })
      console.log(dependentNewFieldDiv)
      // setMode()
      break
    }
    default: {
      fieldName = 'income'

      const incomeFieldContainer = document.getElementById(fieldName)
      const btnID = 'profile-' + fieldName + '-btn'
      const incomeFieldBtn = document.getElementById(btnID)
      incomeFieldContainer.removeChild(incomeFieldBtn)

      const incomeNewFieldDiv = new Income({ target: document.getElementById('income') })
      console.log(incomeNewFieldDiv)
      // setMode()
      break
    }
  }
}
