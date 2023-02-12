const express = require('express')
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
    var data1;
   // spawn new child process to call the python script

   
   const python = spawn('python3', ['crop recommendation.py',20,30,40,30,10,8,100]);
   // collect data from script
   python.stdout.on('data', (data)=> {
      data1 = data.toString();
      console.log(data1);
   });
   // in close event we are sure that stream from child process is closed
   python.on('close', (code) => {
      console.log(`child process close all stdio with code ${code}`);
      // send data to browser
      res.send(data1)
   });
})

app.listen(port,console.log("port is running....."))