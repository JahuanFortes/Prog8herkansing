//#region imports/vars
import "./style.css";
import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import jsonData from "../../preparing-data/src/data.json";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const trainModelBtn = document.getElementById("trainModelBtn");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let image = document.querySelector("#myimage");
let isTrained = false;

ml5.setBackend("webgl");
const options = {
  // inputs: 63,
  task: "classification",
  debug: true,
  layers: [
    {
      type: "dense",
      units: 32,
      activation: "relu",
    },
    {
      type: "dense",
      units: 32,
      activation: "relu",
    },
    {
      type: "dense",
      units: 32,
      activation: "relu",
    },
    {
      type: "dense",
      activation: "softmax",
    },
  ],
};

const nn = ml5.neuralNetwork(options);

const randomData = jsonData.sort(() => Math.random() - 0.5);
const train = randomData.slice(0, Math.floor(randomData.length * 0.8));
const test = randomData.slice(Math.floor(randomData.length * 0.8) + 1);

addDataToNeuralNetwork();
//#endregion imports
// startTraining();

trainModelBtn.addEventListener("click", () => startTraining());
// /********************************************************************
// // CREATE THE POSE DETECTOR
// ********************************************************************/
// const createHandLandmarker = async () => {
//   const vision = await FilesetResolver.forVisionTasks(
//     "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
//   );
//   handLandmarker = await HandLandmarker.createFromOptions(vision, {
//     baseOptions: {
//       modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
//       delegate: "GPU",
//     },
//     runningMode: "VIDEO",
//     numHands: 2,
//   });
//   console.log("model loaded, you can start webcam");

//   enableWebcamButton.addEventListener("click", (e) => enableCam(e));
//   logButton.addEventListener("click", (e) => logAllHands(e));
// };
// /********************************************************************
// // START THE WEBCAM
// ********************************************************************/
// async function enableCam() {
//   webcamRunning = true;
//   try {
//     const stream = await navigator.mediaDevices.getUserMedia({
//       video: true,
//       audio: false,
//     });
//     video.srcObject = stream;
//     video.addEventListener("loadeddata", () => {
//       canvasElement.style.width = video.videoWidth;
//       canvasElement.style.height = video.videoHeight;
//       canvasElement.width = video.videoWidth;
//       canvasElement.height = video.videoHeight;
//       document.querySelector(".videoView").style.height =
//         video.videoHeight + "px";
//       predictWebcam();
//     });
//   } catch (error) {
//     console.error("Error accessing webcam:", error);
//   }
// }

// /********************************************************************
// // START PREDICTIONS
// ********************************************************************/
// async function predictWebcam() {
//   results = await handLandmarker.detectForVideo(video, performance.now());

//   let hand = results.landmarks[0];

//   if (hand) {
//     let thumb = hand[4];
//     image.style.transform = `translate(${
//       video.videoWidth - thumb.x * video.videoWidth
//     }px, ${thumb.y * video.videoHeight}px)`;
//     // train();
//   }

//   canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//   for (let hand of results.landmarks) {
//     drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
//       color: "#00FF00",
//       lineWidth: 5,
//     });
//     drawUtils.drawLandmarks(hand, {
//       radius: 4,
//       color: "#FF0000",
//       lineWidth: 2,
//     });
//   }

//   if (webcamRunning) {
//     window.requestAnimationFrame(predictWebcam);
//   }
// }
// /********************************************************************
// // LOG HAND COORDINATES IN THE CONSOLE
// ********************************************************************/
// function logAllHands() {
//   const hand = results.landmarks;
//   hand.forEach((hand) => {
//     const mdArr = hand
//       .map((landmark) => [landmark.x, landmark.y, landmark.z])
//       .flat();
//     console.log(mdArr);
//   });

//   // console.log(mdArr);
// }

// function setup() {
//   for (let i = 0; i < data.length; i++) {
//     const landmark = data[i];
//     let inputs = [landmark.x, landmark.y, landmark.z];
//   }
// }

function addDataToNeuralNetwork() {
  for (let pose of train) {
    nn.addData(pose.points, { label: pose.label });
  }
}

//#region Training/Accuracy
function startTraining() {
  try {
    nn.normalizeData();

    const trainingOptions = {
      epochs: 40,
      learningRate: 0.3,
      hiddenUnits: 16,
    };

    nn.train(trainingOptions, () => finishedTraining());
  } catch (error) {
    console.log(error);
  }
}

async function finishedTraining() {
  nn.save(
    "model",
    () => console.log("yay!"),
    () => runAccuracyTest()
  );
  runAccuracyTest();
}

async function runAccuracyTest() {
  let correctPose = 0;
  for (const pose of test) {
    const prediction = await nn.classify(pose.points);

    // als prediction === pose.label
    if (prediction[0].label === pose.label) {
      correctPose++;

      console.log("klopt");
    }
  }
  const accuracy = (correctPose / test.length) * 100;
  console.log(accuracy);
}

//#endregion imports

/********************************************************************
// START THE APP
********************************************************************/

// if (navigator.mediaDevices?.getUserMedia) {
//   createHandLandmarker();
// }
