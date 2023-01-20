// Purpose: demonstrate how callbacks work

//Example 1: "A callback function is a function passed into another function as an argument, 
// which is then invoked inside the outer function to complete some kind of routine or action."
// Source: https://developer.mozilla.org/en-US/docs/Glossary/Callback_function

function greeting(name) {
  alert('Hello ' + name);
}

function processUserInput(callback) {
  var name = prompt('Please enter your name.');
  callback(name);
}

processUserInput(greeting);

// Example 2: adapted from Andrew Mead's Udemy course titled "The Complete Node.js Developer course" 
// Takeaway: the sum of the a and b arguments will be printed after the timeout
console.log('Start');

const add = (a, b, callback) => {
    setTimeout(() => {
        callback(a+b);
    }, 2000)
}

add(1,4, (sum) => {
    console.log(sum); // this will print 5
})


// Example 2: Adapted from the Node docs: 
// https://nodejs.org/en/docs/guides/timers-in-node/
// Takeaway: a third argument can be passed in that is accessed via myFun

function myFunc(arg) {
  console.log(`arg was => ${arg}`);
}

setTimeout(myFunc, 1500, 'funky'); // outputs "arg was => funky"

