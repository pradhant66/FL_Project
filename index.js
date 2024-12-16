const express = require('express');

const app = express();
var globalParams = {};
// Define a route handler for the `/getparams` endpoint
app.get('/getparams', (req, res) => {
  // Get the query parameters from the request
//   const { name, age } = req.query;

//   // Build a response object
//   const response = {
//     name,
//     age,
//   };

  // Send the response back to the client
  res.json({globalParams : globalParams});
});
app.post('/setparams', (req, res) => {
  // Get the query parameters from the request
//   const { name, age } = req.query;

//   // Build a response object
//   const response = {
//     name,
//     age,
//   };
console.log(req.query);
aggregateParams();
  // Send the response back to the client
  res.json("noce");
});

function aggregateParams(){
  console.log("hihi")
}

// Start the server on port 3000
app.listen(3001, () => {
  console.log('Server started on port 3001');
});