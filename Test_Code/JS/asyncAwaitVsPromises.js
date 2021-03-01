// Purposes: demonstrate the difference between async await vs promise chaining
// Examples adapted from Andrew Mead's Udemy course: The Complete NodeJS Developer

// Example 1: Create add() function and chain promises
// The key takeaway from this example is how confusing the nested syntax can be
const add = (a, b) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve(a+b);
        }, 2000)
    })
}

add(1,1).then((sum) =>{
    console.log(sum);
    return add(sum, 4)
}).then((sum2) => {
    console.log(sum2);
}).catch((e) => {
    console.log(e)
})

// Example 2: using the asycn await keywords we can clean up the syntax
// The print stmts demonstrate the order the code is executed
const doWork = async () => {
    const sum = await add(1,1);
    //console.log('doWork sum: ' + sum); 
    const sum2 = await add(sum, 1);
    //console.log('doWork sum2: ' + sum2);
    const sum3 = await add(sum2, 1);
    //console.log('doWork sum3: ' + sum3);
    return sum3;
};

doWork().then((result) => {
    console.log('result', result)
}).catch((e) => {
    console.log('e', e);
})
