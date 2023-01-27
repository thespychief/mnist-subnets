/* eslint-disable no-plusplus */
/* eslint-disable no-console */
/* eslint-disable no-restricted-syntax */
const fs = require('fs');
const Readline = require('readline');
const lodash = require('lodash');
const cliProgress = require('cli-progress');
const StitchML = require('stitch-ml');

const getNumLinesInFile = async (file) => {
  const readInterface = Readline.createInterface({
    input: fs.createReadStream(file),
  });

  let lineCount = 0;
  // eslint-disable-next-line no-unused-vars
  for await (const line of readInterface) {
    lineCount++;
  }
  return lineCount;
};

(async () => {
  const subnetCount = 4;
  const subnetDimension = 14;
  const inputDimension = 28;
  const trainingFile = './data/mnist_train.ndjson';
  const testFile = './data/mnist_test.ndjson';
  const trainingFileLineCount = await getNumLinesInFile(trainingFile);
  const testFileLineCount = await getNumLinesInFile(testFile);

  const readInterface = Readline.createInterface({
    input: fs.createReadStream(trainingFile),
  });

  console.log('Building subnet training files...');

  for (let i = 0; i < subnetCount; i++) {
    fs.writeFileSync(`./data/tmp/${i}.ndjson`, '');
  }

  const progressBar = new cliProgress.SingleBar(
    {}, cliProgress.Presets.shades_classic,
  );
  progressBar.start(trainingFileLineCount, 0);

  let currentLine = 0;
  for await (const line of readInterface) {
    const point = JSON.parse(line);
    const inputMatrix = lodash.chunk(point.input, inputDimension);

    const convolutions = StitchML.Matrix.convolve(
      inputMatrix, subnetDimension,
    );

    for (let j = 0; j < convolutions.length; j++) {
      fs.appendFileSync(`./data/tmp/${j}.ndjson`, `${JSON.stringify({
        input: lodash.flattenDeep(convolutions[j]),
        output: point.output,
      })}\n`);
    }

    currentLine++;
    progressBar.update(currentLine);
  }
  progressBar.update(trainingFileLineCount);
  progressBar.stop();

  console.log('Training subnets...');

  const networks = [];
  for (let i = 0; i < subnetCount; i++) {
    networks.push({
      network: { id: i, structure: [196, 25, 10], activation: 'Sigmoid' },
      training: {},
      file: `./data/tmp/${i}.ndjson`,
    });
  }

  console.time();
  const trainedSubnets = await StitchML.trainInParallelFromFiles(networks);
  console.timeEnd();

  const orderedSubnets = lodash.orderBy(trainedSubnets, ['id'], ['asc']);
  const subnets = orderedSubnets.map((subnet) => new StitchML.Network(subnet));

  const readInterface2 = Readline.createInterface({
    input: fs.createReadStream(trainingFile),
  });

  console.log('Building StitchNet training data...');

  const progressBar2 = new cliProgress.SingleBar(
    {}, cliProgress.Presets.shades_classic,
  );
  progressBar2.start(trainingFileLineCount, 0);

  const stitchTrainingData = [];
  let currentLine2 = 0;
  for await (const line of readInterface2) {
    const point = JSON.parse(line);
    const inputMatrix = lodash.chunk(point.input, inputDimension);

    const convolutions = StitchML.Matrix.convolve(
      inputMatrix, subnetDimension,
    );

    const outputs = [];
    for (let j = 0; j < convolutions.length; j++) {
      outputs.push(
        subnets[j].predict(lodash.flattenDeep(convolutions[j])),
      );
    }

    stitchTrainingData.push({
      input: lodash.flatten(outputs),
      output: point.output,
    });

    currentLine2++;
    progressBar2.update(currentLine2);
  }
  progressBar2.update(trainingFileLineCount);
  progressBar2.stop();

  const stitchNet = new StitchML.Network({
    activation: 'Sigmoid',
    structure: [40, 20, 10],
  });

  console.log('Training StitchNet...');

  console.time();
  await stitchNet.train({
    data: stitchTrainingData,
  });
  console.timeEnd();

  const readInterface3 = Readline.createInterface({
    input: fs.createReadStream(testFile),
  });

  console.log('Evaluating...');

  const progressBar3 = new cliProgress.SingleBar(
    {}, cliProgress.Presets.shades_classic,
  );
  progressBar3.start(testFileLineCount, 0);

  let numCorrect = 0;
  let numIncorrect = 0;
  for await (const line of readInterface3) {
    const point = JSON.parse(line);
    const inputMatrix = lodash.chunk(point.input, inputDimension);

    const convolutions = StitchML.Matrix.convolve(
      inputMatrix, subnetDimension,
    );

    const outputs = [];
    for (let j = 0; j < convolutions.length; j++) {
      outputs.push(
        subnets[j].predict(lodash.flattenDeep(convolutions[j])),
      );
    }

    const prediction = stitchNet.predict(lodash.flatten(outputs));

    const isAccurate = lodash.isEqual(
      lodash.indexOf(prediction, lodash.max(prediction)),
      lodash.indexOf(point.output, lodash.max(point.output)),
    );

    // eslint-disable-next-line no-unused-expressions
    isAccurate ? numCorrect++ : numIncorrect++;

    progressBar3.update(numCorrect + numIncorrect);
  }
  progressBar3.update(testFileLineCount);
  progressBar3.stop();

  console.log({
    numCorrect,
    numIncorrect,
    accuracyPrct: numCorrect / (numCorrect + numIncorrect),
  });

  console.log('subnet training times: ', subnets.map(
    (subnet) => subnet.history[0].time,
  ));
  console.log('stitch net training time: ', stitchNet.history[0].time);
})();
