console.log('Start')

      twoSecondTimer = () => {
              console.log('2 second timer')
            }
      setTimeout(twoSecondTimer, 2000);
     
      zeroSecondTimer = () => {
              console.log('0 second timer')
            }
      setTimeout(zeroSecondTimer, 0);
     
      console.log('End')

