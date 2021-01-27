// Purpose: demonstrate how callbacks work


// Example 1: adapted from Andrew Mead's Udemy course titled "The Complete Node.js Developer course" 
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

