import { writable } from 'svelte/store'
/* eslint-env browser */

export const conversationId = writable(-1)

export const modal = writable(null)
export const windowStyle = writable({})
export const darkMode = writable(true)
export const profiles = writable(localStorage.getItem('profiles'))
export const chosenProfile = writable({})
export const showSearchResults = writable(false)
