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
    const grayscaleData = new Array(28 * 28);

    // Convert pixel data to grayscale and normalize to [0, 1]
    for (let i = 0; i < pixelData.length; i += 4) {
        const r = pixelData[i];
        const g = pixelData[i + 1];
        const b = pixelData[i + 2];
        const grayscale = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255;
        grayscaleData[i / 4] = grayscale;
    }

    //console.log(grayscaleData);
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
    console.log(model);
    //await train(model , data);
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
        trainingLabels.push(... Array(splitted[1].length).fill(index));
        testingLabels.push(... Array(splitted[0].length).fill(index));
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