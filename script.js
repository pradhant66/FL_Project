console.log('Hello TensorFlow');
 import {MnistData} from './data.js';

 document.getElementById("nice").addEventListener("click",()=>{console.log(data)})

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {  
  const dataa = new MnistData();
  await dataa.load();
  await showExamples(dataa);
  //await train(model, data);
  //await model.layers[0].getWeights()[0].print();

 // print(model.layers[0].bias.numpy())
  //print(model.layers[0].bias_initializer)
}


// function getModel() {
//     const model = tf.sequential();
    
//     const IMAGE_WIDTH = 28;
//     const IMAGE_HEIGHT = 28;
//     const IMAGE_CHANNELS = 1;  
    
//     // In the first layer of our convolutional neural network we have 
//     // to specify the input shape. Then we specify some parameters for 
//     // the convolution operation that takes place in this layer.
//     model.add(tf.layers.conv2d({
//       inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
//       kernelSize: 5,
//       filters: 8,
//       strides: 1,
//       activation: 'relu',
//       kernelInitializer: 'varianceScaling'
//     }));
  
//     // The MaxPooling layer acts as a sort of downsampling using max values
//     // in a region instead of averaging.  
//     model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
//     // Repeat another conv2d + maxPooling stack. 
//     // Note that we have more filters in the convolution.
//     model.add(tf.layers.conv2d({
//       kernelSize: 5,
//       filters: 16,
//       strides: 1,
//       activation: 'relu',
//       kernelInitializer: 'varianceScaling'
//     }));
//     model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
//     // Now we flatten the output from the 2D filters into a 1D vector to prepare
//     // it for input into our last layer. This is common practice when feeding
//     // higher dimensional data to a final classification output layer.
//     model.add(tf.layers.flatten());
  
//     // Our last layer is a dense layer which has 10 output units, one for each
//     // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
//     const NUM_OUTPUT_CLASSES = 10;
//     model.add(tf.layers.dense({
//       units: NUM_OUTPUT_CLASSES,
//       kernelInitializer: 'varianceScaling',
//       activation: 'softmax'
//     }));
  
    
//     // Choose an optimizer, loss function and accuracy metric,
//     // then compile and return the model
//     const optimizer = tf.train.adam();
//     model.compile({
//       optimizer: optimizer,
//       loss: 'categoricalCrossentropy',
//       metrics: ['accuracy'],
//     });
  
//     return model;
//   }

function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;  
  
 
  model.add(tf.layers.dense({
      units: 200,
      inputShape: 784,
      activation: 'relu',
      name: 'dense_layer_1'
    }));
    // Add more layers to the model...

    // model.add(tf.layers.flatten({inputShape: [28,28] , name: 'layer_1'}));
    // // Add more layers to the model...
    

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

  const optimizer = tf.train.sgd(lr, momentum, undefined, decay);

 
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
  
    // const [trainXs, trainYs] = tf.tidy(() => {
    //   const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    //   return [
    //     d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
    //     d.labels
    //   ];
    // });
  
    // const [testXs, testYs] = tf.tidy(() => {
    //   const d = data.nextTestBatch(TEST_DATA_SIZE);
    //   return [
    //     d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
    //     d.labels
    //   ];
    // });
    var temp = GeneratetestAndTrainingSets(data , 10);
    //console.log(temp.shuffledTestingImages);
    const aa = temp.shuffledTrainingImages.flat();
    const bb = temp.shuffledTrainingLabels.flat();
    const cc = temp.shuffledTestingImages.flat();
    const dd = temp.shuffledTestingLabels.flat();
    //console.log(aa);
    const xs = tf.tensor(aa, [temp.shuffledTrainingImages.length, 784]);
    const ys = tf.tensor(bb, [temp.shuffledTrainingImages.length, 10]);
    const xt = tf.tensor(cc, [temp.shuffledTestingImages.length, 784]);
    const yt = tf.tensor(dd, [temp.shuffledTestingLabels.length, 10]);
var batch_size = Math.round(temp.shuffledTrainingImages.length/10);
console.log(batch_size)
    // console.log(aa);
    // console.log(aa.length);
    // console.log("train");
    // console.log(bb);
    //   const xs = tf.tensor2d(aa , [temp.shuffledTrainingImages.length, 784]);
    //  const ys = tf.tensor2d(bb , [temp.shuffledTrainingLabels.length, 784]);
    //  xs.reshape(xs.length,28,28,1);
    // //const xs = temp.shuffledTrainingImages;
    // //xs.reshape([xs.length,28,28,1]);
    // //const ys = temp.shuffledTrainingLabels;
    // //console.log(xs)
    // const dim1 = xs.length; // 2
    // const dim2 = xs[0].length; // 2
    // const dim3 = xs[0][0].length; // 2

    //console.log(dim1, dim2, dim3);
     return model.fit(xs, ys, {
       batchSize: Math.max(512,batch_size),
       validationData: [xt,yt],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
    console.log('trained')
    
  }

const model = getModel();
tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  
document.addEventListener('DOMContentLoaded', run);


////////

// import { train } from "./script.js";
let data = [];
for (let i = 0; i < 10; i++) {
  data.push([]);
}
var data0 = [];
async function getArrayOfImage(url , idx){

    const image = new Image();
    image.src = url;

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const context = canvas.getContext('2d');

    image.onload = function() {
    context.drawImage(image, 0, 0, 28, 28);
    const imageData = context.getImageData(0, 0, 28, 28);
    const pixelData = imageData.data;
    const grayscaleData = new Float32Array(28 * 28);

    // Convert pixel data to grayscale and normalize to [0, 1]
    for (let i = 0; i < pixelData.length; i += 4) {
        const r = pixelData[i];
        const g = pixelData[i + 1];
        const b = pixelData[i + 2];
        const grayscale = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255;
        grayscaleData[i / 4] = grayscale;
    }
// Convert the pixel values to a 2D array
// const pixelArray = [];
// for (let i = 0; i < pixelData.length; i += 4) {
//   const grayValue = (pixelData[i] + pixelData[i + 1] + pixelData[i + 2]) / 3; // Convert to grayscale
//   pixelArray.push((grayValue / 255)); // Normalize to 0-1 and round to 0 or 1
// }
// const result = [];
// for (let i = 0; i < 28; i++) {
//   result.push(pixelArray.slice(i * 28, i * 28 + 28));
// }

//console.log(result); // Output the resulting 2D array
    //console.log(grayscaleData);
   //data[idx].push(result);
   data[idx].push(grayscaleData);
   //return grayscaleData;

};

}
const inputList = document.querySelectorAll("input")
var imagesArray = [];
for (let i = 0; i < 10; i++) {
  imagesArray.push([]);
}
inputList.forEach((input , index)=>{
input.addEventListener("change", () => {
    const files = input.files
    for (let i = 0; i < files.length; i++) {
       imagesArray[index].push(files[i])
    //console.log(files[i]);
    }
      populateImageArray(index);
  })
})



  async function populateImageArray(idx) {
    imagesArray[idx].forEach((image) => {
     getArrayOfImage(`${URL.createObjectURL(image)}` , idx);
    // console.log(curr);
    // data[idx].push(curr);
})
  }

function splitArrayByPercentage(arr , percentage){
    var factor = 100/percentage;
    const shuffledArray = arr.sort(() => 0.5 - Math.random());
    const splitIndex = Math.floor(shuffledArray.length /factor);
    const array1 = shuffledArray.slice(0, splitIndex);
    const array2 = shuffledArray.slice(splitIndex);
    return [array1 , array2];
}

const generateButton = document.getElementById("generate");
generateButton.addEventListener("click" , async ()=>{
    //console.log(GeneratetestAndTrainingSets(data, 10))
    //console.log(model);
    await train(model , data);
    await model.layers[0].getWeights()[0].print();
    console.log("hello")
})


function GeneratetestAndTrainingSets(data , percentageOfTests ){
    var trainingImages = [];
    var trainingLabels = [];
    var testingImages  = [];
    var testingLabels  = [];

    data.forEach((imagesArray , index) => {
        var splitted = splitArrayByPercentage(imagesArray , percentageOfTests);
        trainingImages.push(...splitted[1]);
        testingImages.push(...splitted[0]);
        var iarr = [];
        for(var i=0;i<10;i++){
          if(i == index){
            iarr.push(1);
          }else{
            iarr.push(0);
          }
        }
        trainingLabels.push(... Array(splitted[1].length).fill(iarr));
        testingLabels.push(... Array(splitted[0].length).fill(iarr));
    });



    const indices = Array.from({length: trainingImages.length}, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    const shuffledTrainingImages = indices.map(i => trainingImages[i]);
    const shuffledTrainingLabels = indices.map(i => trainingLabels[i]);

    const indices2 = Array.from({length: testingImages.length}, (_, i) => i);
    for (let i = indices2.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices2[i], indices2[j]] = [indices2[j], indices2[i]];
      }
    const shuffledTestingImages = indices2.map(i => testingImages[i]);
    const shuffledTestingLabels = indices2.map(i => testingLabels[i]);

    return {shuffledTrainingImages,shuffledTrainingLabels,shuffledTestingImages,shuffledTestingLabels};
}



var d = [
    [[1,2,3,4],[13,25,34,44],[11,22,33,44],[5,6,7,8]],
    [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
]