import { initializeApp } from 'firebase/app'
import { getAnalytics } from 'firebase/analytics'

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: 'AIzaSyBRubqun02aPtly4KDrBVbAXBmoAXs1OZw',
  authDomain: 'fingenie-23c88.firebaseapp.com',
  projectId: 'fingenie-23c88',
  storageBucket: 'fingenie-23c88.appspot.com',
  messagingSenderId: '86647911124',
  appId: '1:86647911124:web:59693dbce2c147692b0141',
  measurementId: 'G-YPX5SDCWJR'
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)
const analytics = getAnalytics(app)

export default analytics
