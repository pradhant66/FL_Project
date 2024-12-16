
function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
   
    model.add(tf.layers.dense({
        units: 200,
        inputShape: [784],
        activation: 'relu',
        name: 'dense_layer_1'
      }));
      

    model.add(tf.layers.dense({
        units: 200, // units for the hidden layer
        activation: 'relu',
        name: 'hidden_layer'
      }));

    const NUM_OUTPUT_CLASSES = 10;

    
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        activation: 'softmax',
        name: 'output'

    }));



    const lr = 0.01;
    const comms_round = 20;

    const decay = lr / comms_round;
    const momentum = 0.9;

    const optimizer = tf.train.sgd(learningRate, momentum, undefined, decay);

   
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }


  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  

    const epochs = 1;
    const verbose = 0;

    return model.fit(clients_batched[client], {
    epochs: epochs,
    verbose: verbose
    });


    // return model.fit(trainXs, trainYs, {
    //   batchSize: BATCH_SIZE,
    //   validationData: [testXs, testYs],
    //   epochs: 10,
    //   shuffle: true,
    //   callbacks: fitCallbacks
    // });
  }