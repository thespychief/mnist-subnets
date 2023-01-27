/* eslint-disable no-console */
const lodash = require('lodash');
const StitchML = require('stitch-ml');

(async () => {
  const network = new StitchML.Network({
    activation: 'Sigmoid',
    structure: [784, 100, 10],
  });

  console.log('Training...');

  await network.train({
    file: './data/mnist_train.ndjson',
    epochs: 1,
    learningRate: 0.1,
  });

  console.log('Evaluating...');

  const results = await network.evaluateFromFileStream({
    file: './data/mnist_test.ndjson',
    func: ({ output, prediction }) => lodash.isEqual(
      lodash.indexOf(prediction, lodash.max(prediction)),
      lodash.indexOf(output, lodash.max(output)),
    ),
  });

  console.log();
  console.log(results);
  console.log('training time: ', network.history[0].time);
})();
