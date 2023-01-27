const http = require('https');
const fs = require('fs');

if (!fs.existsSync('./data')) {
  fs.mkdirSync('./data');
  fs.mkdirSync('./data/tmp');
}

const trainFile = fs.createWriteStream('./data/mnist_train.ndjson');
http.get(
  'https://thespychief.nyc3.cdn.digitaloceanspaces.com/mnist_train.ndjson',
  (response) => {
    response.pipe(trainFile);

    trainFile.on('finish', () => {
      trainFile.close();
      console.log('Training File Download Completed');
    });
  },
);

const testFile = fs.createWriteStream('./data/mnist_test.ndjson');
http.get(
  'https://thespychief.nyc3.cdn.digitaloceanspaces.com/mnist_test.ndjson',
  (response) => {
    response.pipe(testFile);

    testFile.on('finish', () => {
      testFile.close();
      console.log('Test File Download Completed');
    });
  },
);
